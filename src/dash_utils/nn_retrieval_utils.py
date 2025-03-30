from typing import List, Optional
import sys
import torch
import faiss
import json
from io import BytesIO
from PIL import Image
import base64
import os
import numpy as np
import math
import urllib.request
from clip_retrieval.clip_back import load_clip_indices, KnnService, ClipOptions, KNN_INDEX_TIME, normalized, TEXT_CLIP_INFERENCE_TIME, TEXT_PREPRO_TIME, IMAGE_PREPRO_TIME, IMAGE_CLIP_INFERENCE_TIME
from multiprocessing import Pool, Manager

from .dreamsim_utils import load_dreamsim_model, compute_dreamsim_embeddings
from .utils import get_nn_infos_filepath, get_prompt_infos_filepath, get_nn_filepath

#Modified version that returns embeddings for duplicate removal
class KnnServiceReturnEmbeddings(KnnService):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def knn_search(
        self, query, modality, num_result_ids, clip_resource, deduplicate, use_safety_model, use_violence_detector
    ):
        """compute the knn search"""

        image_index = clip_resource.image_index
        text_index = clip_resource.text_index
        if clip_resource.metadata_is_ordered_by_ivf:
            ivf_old_to_new_mapping = clip_resource.ivf_old_to_new_mapping

        index = image_index if modality == "image" else text_index

        with KNN_INDEX_TIME.time():
            if clip_resource.metadata_is_ordered_by_ivf:
                previous_nprobe = faiss.extract_index_ivf(index).nprobe
                if num_result_ids >= 100000:
                    nprobe = math.ceil(num_result_ids / 3000)
                    params = faiss.ParameterSpace()
                    params.set_index_parameters(index, f"nprobe={nprobe},efSearch={nprobe*2},ht={2048}")
            distances, indices, embeddings = index.search_and_reconstruct(query, num_result_ids)
            if clip_resource.metadata_is_ordered_by_ivf:
                results = np.take(ivf_old_to_new_mapping, indices[0])
            else:
                results = indices[0]
            if clip_resource.metadata_is_ordered_by_ivf:
                params = faiss.ParameterSpace()
                params.set_index_parameters(index, f"nprobe={previous_nprobe},efSearch={previous_nprobe*2},ht={2048}")
        nb_results = np.where(results == -1)[0]

        if len(nb_results) > 0:
            nb_results = nb_results[0]
        else:
            nb_results = len(results)
        result_indices = results[:nb_results]
        result_distances = distances[0][:nb_results]
        result_embeddings = embeddings[0][:nb_results]
        result_embeddings = normalized(result_embeddings)
        local_indices_to_remove = self.post_filter(
            clip_resource.safety_model,
            result_embeddings,
            deduplicate,
            use_safety_model,
            use_violence_detector,
            clip_resource.violence_detector,
        )
        indices_to_remove = set()
        for local_index in local_indices_to_remove:
            indices_to_remove.add(result_indices[local_index])
        indices = []
        distances = []
        embeddings = []
        for ind, distance, emb in zip(result_indices, result_distances, result_embeddings):
            if ind not in indices_to_remove:
                indices_to_remove.add(ind)
                indices.append(ind)
                distances.append(distance)
                embeddings.append(emb)

        return distances, indices, embeddings

    def query(
        self,
        text_input=None,
        image_input=None,
        image_url_input=None,
        embedding_input=None,
        modality="image",
        num_images=100,
        num_result_ids=100,
        indice_name=None,
        use_mclip=False,
        deduplicate=True,
        use_safety_model=False,
        use_violence_detector=False,
        aesthetic_score=None,
        aesthetic_weight=None,
    ):
        """implement the querying functionality of the knn service: from text and image to nearest neighbors"""

        if text_input is None and image_input is None and image_url_input is None and embedding_input is None:
            raise ValueError("must fill one of text, image and image url input")
        if indice_name is None:
            indice_name = next(iter(self.clip_resources.keys()))

        clip_resource = self.clip_resources[indice_name]

        query = self.compute_query(
            clip_resource=clip_resource,
            text_input=text_input,
            image_input=image_input,
            image_url_input=image_url_input,
            embedding_input=embedding_input,
            use_mclip=use_mclip,
            aesthetic_score=aesthetic_score,
            aesthetic_weight=aesthetic_weight,
        )
        distances, indices, embeddings = self.knn_search(
            query,
            modality=modality,
            num_result_ids=num_result_ids,
            clip_resource=clip_resource,
            deduplicate=deduplicate,
            use_safety_model=use_safety_model,
            use_violence_detector=use_violence_detector,
        )
        if len(distances) == 0:
            return []
        results = self.map_to_metadata(
            indices, distances, num_images, clip_resource.metadata_provider, clip_resource.columns_to_return
        )

        assert len(embeddings) == len(results)
        for res, emb in zip(results, embeddings):
            res['embedding'] = emb

        return results

    def compute_query(
        self,
        clip_resource,
        text_input,
        image_input,
        image_url_input,
        embedding_input,
        use_mclip,
        aesthetic_score,
        aesthetic_weight,
    ):
        """compute the query embedding"""
        import torch  # pylint: disable=import-outside-toplevel

        if text_input is not None and text_input != "":
            if use_mclip:
                with TEXT_CLIP_INFERENCE_TIME.time():
                    query = normalized(clip_resource.model_txt_mclip(text_input))
            else:
                with TEXT_PREPRO_TIME.time():
                    text = clip_resource.tokenizer([text_input]).to(clip_resource.device)
                with TEXT_CLIP_INFERENCE_TIME.time():
                    with torch.no_grad():
                        text_features = clip_resource.model.encode_text(text)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    query = text_features.cpu().to(torch.float32).detach().numpy()
        elif image_input is not None or image_url_input is not None:
            if image_input is not None:
                binary_data = base64.b64decode(image_input)
                img_data = BytesIO(binary_data)
            elif image_url_input is not None:
                img_data = download_image(image_url_input)
            with IMAGE_PREPRO_TIME.time():
                from PIL import PngImagePlugin
                LARGE_ENOUGH_NUMBER = 100
                PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

                img = Image.open(img_data)
                prepro = clip_resource.preprocess(img).unsqueeze(0).to(clip_resource.device)
            with IMAGE_CLIP_INFERENCE_TIME.time():
                with torch.no_grad():
                    image_features = clip_resource.model.encode_image(prepro)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                query = image_features.cpu().to(torch.float32).detach().numpy()
        elif embedding_input is not None:
            query = np.expand_dims(np.array(embedding_input).astype("float32"), 0)

        if clip_resource.aesthetic_embeddings is not None and aesthetic_score is not None:
            aesthetic_embedding = clip_resource.aesthetic_embeddings[aesthetic_score]
            query = query + aesthetic_embedding * aesthetic_weight
            query = query / np.linalg.norm(query)

        return query

# accumulate
class NNInfo():
    def __init__(self, id, url, caption, idx, nn_similarity, embedding):
        self.id = id
        self.url = url
        self.caption = caption
        self.idcs_similarity = {idx: nn_similarity}
        self.embedding = embedding

    def update(self, idx, similarity):
        self.idcs_similarity[idx] = similarity


def resize_longest_edge(img: Image, max_size=1024):
    if img.mode not in ('L', 'RGB'):
        img = img.convert('RGB')

    width, height = img.size

    if max(width, height) <= max_size:
        return img

    factor = max_size / max(width, height)
    new_w = int(width * factor)
    new_h = int(height * factor)

    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    return img


def download_image(url, path, min_size=64, max_size=1024) -> bool:
    try:
        #already downloaded
        if os.path.isfile(path):
            return True

        from PIL import PngImagePlugin
        LARGE_ENOUGH_NUMBER = 100
        PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

        timeout = 5
        user_agent_string = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:72.0) Gecko/20100101 Firefox/72.0"
        request = urllib.request.Request(url, data=None, headers={"User-Agent": user_agent_string})
        img_stream = None
        with urllib.request.urlopen(request, timeout=timeout) as r:
            img_stream = BytesIO(r.read())

        nn_img = Image.open(img_stream)
        width, height = nn_img.size
        if min(width, height) < min_size:
            return False

        nn_img = resize_longest_edge(nn_img, max_size=max_size)
        nn_img.save(path)
        return True
    except KeyboardInterrupt:
        if img_stream is not None:
            img_stream.close()
        sys.exit(0)
    except Exception as e:
        if img_stream is not None:
            img_stream.close()
        #print(f'Warning could not download {url} - {e}')
        return False


def compute_similarity_embedding_to_success_embeddings(embedding, success_embeddings):
    if success_embeddings:
        similarities_to_success = np.dot(np.stack(success_embeddings), embedding[:, None])
        max_similarity = np.max(similarities_to_success)
        return max_similarity
    else:
        return 0.


class Retriever:
    def __init__(self, num_workers, index_file='indices.json', clip_options=None,
                 clip_similarity_threshold: Optional[float] = None,
                 dreamsim_similarity_threshold: Optional[float] = None,
                 dreamsim_batch_size: int = 64,
                 ):
        self.pool = Pool(processes=num_workers)
        self.manager = Manager()
        if clip_options is None:
            clip_options = ClipOptions(
                indice_folder='/mnt/datasets/LAION5B_knn/Laion5B_H14',
                clip_model='open_clip:ViT-H-14',
                enable_hdf5=False,
                enable_faiss_memory_mapping=True,
                columns_to_return=["url", "caption"],
                reorder_metadata_by_ivf_index=False,
                enable_mclip_option=False,  # True,
                use_arrow=True,
                provide_safety_model=False,
                use_jit=True,
                provide_violence_detector=False,
                provide_aesthetic_embeddings=False
            )
        clip_resources = load_clip_indices(index_file, clip_options)
        self.knn_service = KnnServiceReturnEmbeddings(clip_resources=clip_resources)

        self.clip_similarity_threshold = clip_similarity_threshold
        self.dreamsim_similarity_threshold = dreamsim_similarity_threshold
        self.dreamsim_batch_size = dreamsim_batch_size
        if dreamsim_similarity_threshold is not None:
            self.dreamsim_model, self.dreamsim_preprocess = load_dreamsim_model(device=torch.device('cuda'))
        else:
            self.dreamsim_model = self.dreamsim_preprocess = None

    def download_from_text(self, text_prompts, num_images, num_requests, result_dir, resume=False):
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]

        nn_results = {}
        idx_to_prompt = {}
        for idx, prompt in enumerate(text_prompts):
            results = self.knn_service.query(text_input=prompt, modality='image', indice_name='laion5B-H-14',
                                        deduplicate=True,
                                        num_images=num_requests, num_result_ids=num_requests)
            nn_results[idx] = results
            idx_to_prompt[idx] = prompt

        prompt_info_dict = {
            'prompt_to_idx': idx_to_prompt,
        }
        with open(get_prompt_infos_filepath(result_dir), 'w') as fp:
            json.dump(prompt_info_dict, fp)

        return self._joint_download(nn_results, num_images, result_dir, resume=resume)

    def download_from_images(self, image_paths, num_images, num_requests, result_dir, resume=False):
        nn_results = {}
        idx_to_prompt = {}
        for idx, image_path in enumerate(image_paths):
            with open(os.path.join(image_path), "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())
                image = str(encoded_string.decode("utf-8"))
            results = self.knn_service.query(image_input=image, modality='image', indice_name='laion5B-H-14', deduplicate=True,
                                        num_images=num_requests, num_result_ids=num_requests)
            nn_results[idx] = results
            idx_to_prompt[idx] = image_path

        prompt_info_dict = {
            'prompt_to_idx': idx_to_prompt,
        }
        with open(get_prompt_infos_filepath(result_dir), 'w') as fp:
            json.dump(prompt_info_dict, fp)

        return self._joint_download(nn_results, num_images, result_dir, resume=resume)

    def _get_similarity_checkers(self):
        id_to_clip_embeddings = {}
        accepted_clip_embeddings = []
        if self.dreamsim_similarity_threshold is not None:
            id_to_dreamsim_embeddings = {}
            accepted_dreamsim_embeddings = []
        else:
            id_to_dreamsim_embeddings = None
            accepted_dreamsim_embeddings = None

        #this can be run to immediately check before download using clip embedings only
        def pre_similarity_checker(nn_info: NNInfo):
            if self.clip_similarity_threshold is not None:
                max_clip_similarity = compute_similarity_embedding_to_success_embeddings(nn_info.embedding, accepted_clip_embeddings)
                if max_clip_similarity > self.clip_similarity_threshold:
                    return False
            return True

        #this can be run after images are downloaded and uses dreamsim and clip
        def post_similarity_checker(nn_info: NNInfo, dreamsim_embedding=None):
            if self.clip_similarity_threshold is not None:
                max_clip_similarity = compute_similarity_embedding_to_success_embeddings(nn_info.embedding, accepted_clip_embeddings)
                if max_clip_similarity > self.clip_similarity_threshold:
                    return False
            if self.dreamsim_similarity_threshold is not None:
                assert dreamsim_embedding is not None
                max_dreamsim_similarity = compute_similarity_embedding_to_success_embeddings(dreamsim_embedding, accepted_dreamsim_embeddings)
                if max_dreamsim_similarity > self.dreamsim_similarity_threshold:
                    return False

            #image passed both checks and can be added to list to check against next
            id_to_clip_embeddings[nn_info.id] = nn_info.embedding
            accepted_clip_embeddings.append(nn_info.embedding)
            if self.dreamsim_similarity_threshold is not None:
                id_to_dreamsim_embeddings[nn_info.id] = dreamsim_embedding
                accepted_dreamsim_embeddings.append(dreamsim_embedding)

            return True

        return pre_similarity_checker, post_similarity_checker, id_to_clip_embeddings, id_to_dreamsim_embeddings

    def _compute_post_download_dreamsim_embeddings(self, pool_results, to_do, result_dir, old_file_infos=None):
        lin_idx_to_id = {}
        nn_paths = []
        id_to_embedding = {}
        for pool_res, nn_info in zip(pool_results, to_do):
            if pool_res:
                nn_path = get_nn_filepath(result_dir, nn_info.id)

                # check if old is usable
                if old_file_infos is not None and nn_info.id in old_file_infos and  old_file_infos[nn_info.id].get('dreamsim_embedding', None) is not None:
                    id_to_embedding[nn_info.id] = old_file_infos[nn_info.id]['dreamsim_embedding']
                else:
                    #prepare for compute
                    lin_idx_to_id[len(nn_paths)] = nn_info.id
                    nn_paths.append(nn_path)

        if not nn_paths:
            return id_to_embedding

        dreamsim_embeddings = compute_dreamsim_embeddings(self.dreamsim_model, self.dreamsim_preprocess, nn_paths,
                                                          batch_size=self.dreamsim_batch_size, device=torch.device('cuda'))
        dreamsim_embeddings = dreamsim_embeddings.detach().cpu().numpy()
        for  lin_idx, id in lin_idx_to_id.items():
            id_to_embedding[id] = dreamsim_embeddings[lin_idx]
        return id_to_embedding

    def _joint_download(self, nn_results_inputs, num_images, result_dir, resume=False):
        nn_infos = []
        laion_id_to_nn_info = {}
        idx_to_nn_infos = {}
        for idx, nn_results in nn_results_inputs.items():
            idx_to_nn_infos[idx] = []
            for result in nn_results:
                result_laion_id = result['id']
                if result_laion_id not in laion_id_to_nn_info:
                    nn_info = NNInfo(id=result_laion_id, url=result['url'], caption=result['caption'],
                                     idx=idx, nn_similarity=result['similarity'], embedding=result['embedding'])
                    laion_id_to_nn_info[result_laion_id] = nn_info
                    idx_to_nn_infos[idx].append(nn_info)
                    nn_infos.append(nn_info)
                else:
                    matching_info = laion_id_to_nn_info[result_laion_id]
                    matching_info.update(idx, result['similarity'])
                    idx_to_nn_infos[idx].append(matching_info)

        missing_per_idx = {idx: num_images for idx in idx_to_nn_infos}
        idx_to_next_position = {idx: 0 for idx in idx_to_nn_infos}
        id_to_success = {}
        pre_similarity_checker, post_similarity_checker, id_to_clip_embeddings, id_to_dreamsim_embeddings = self._get_similarity_checkers()

        nn_info_filepath = get_nn_infos_filepath(result_dir)
        if resume and os.path.isfile(nn_info_filepath):
            old_nn_dict = torch.load(nn_info_filepath)
            old_file_infos = old_nn_dict['file_infos']
        else:
            old_file_infos = None

        while any(num_missing > 0 for num_missing in missing_per_idx.values()):
            to_do = []

            #collect todos
            for idx in missing_per_idx.keys():
                idx_num_to_add = missing_per_idx[idx]
                while (idx_num_to_add > 0):
                    #if exhausted all options
                    if (idx_to_next_position[idx] >= len(idx_to_nn_infos[idx])):
                        missing_per_idx[idx] = 0
                        break

                    next_position = idx_to_next_position[idx]
                    idx_to_next_position[idx] = next_position + 1
                    next_info: NNInfo = idx_to_nn_infos[idx][next_position]

                    #nn has not been attempted
                    if next_info.id not in id_to_success:
                        #check if too similar using precomputed clip embeddings
                        if not pre_similarity_checker(next_info):
                            id_to_success[next_info.id] = False
                            continue
                        to_do.append(next_info)
                        id_to_success[next_info.id] = False
                        idx_num_to_add -= 1

                    ##nn already loaded
                    elif id_to_success[next_info.id]:
                        idx_num_to_add -= 1


            pool_args = []
            for nn_info in to_do:
                nn_path = get_nn_filepath(result_dir, nn_info.id)
                #url, path
                pool_args.append((nn_info.url, nn_path))

            pool_results = self.pool.starmap(download_image, pool_args)

            if self.dreamsim_similarity_threshold is not None:
                new_dreamsim_embeddings = self._compute_post_download_dreamsim_embeddings(pool_results, to_do, result_dir, old_file_infos=old_file_infos)
            else:
                new_dreamsim_embeddings = {}
            for pool_res, nn_info in zip(pool_results, to_do):
                if pool_res:
                    #check if downloaded is too similar to existing images
                    if not post_similarity_checker(nn_info, dreamsim_embedding=new_dreamsim_embeddings.get(nn_info.id, None)):
                        #don't accept this file since too close to downloaded ones
                        id_to_success[nn_info.id] = False
                        try:
                            nn_path = get_nn_filepath(result_dir, nn_info.id)
                            os.remove(nn_path)
                        except Exception as e:
                            print(f'Warning could not delete too similar image: {e}')
                        continue

                    id_to_success[nn_info.id] = True
                    for idx in nn_info.idcs_similarity.keys():
                        missing_per_idx[idx] -= 1

        #save the nn infos
        file_infos = {}
        for info in nn_infos:
            if not id_to_success.get(info.id, False):
                continue

            info_dict = {
                'url': info.url,
                'caption': info.caption,
                'id': info.id,
                'idcs_similarity': info.idcs_similarity,
                'clip_embedding': info.embedding
            }

            if self.dreamsim_similarity_threshold is not None:
                info_dict['dreamsim_embedding'] = id_to_dreamsim_embeddings[info.id]

            file_infos[info.id] = info_dict

        #
        deleted_unused_nns = 0
        for file in os.listdir(result_dir):
            if file.endswith('.png'):
                try:
                    nn_id = int(file.split('.')[0])
                    if nn_id not in file_infos:
                        os.remove(os.path.join(result_dir, file))
                        deleted_unused_nns += 1
                except Exception as e:
                    pass
        print(f'Deleted {deleted_unused_nns} unused NNs')


        #
        source_dicts = {}
        for idx, nn_infos in idx_to_nn_infos.items():
            source_dicts[idx] = []

            for nn_info in nn_infos:
                if len(source_dicts[idx]) >= num_images:
                    break

                if not nn_info.id in file_infos:
                    continue

                source_dicts[idx].append({
                    'id': nn_info.id,
                    'similarity': nn_info.idcs_similarity[idx]
                })

        info_out_dict = {
            'file_infos': file_infos,
            'source_nn_infos': source_dicts
        }
        torch.save(info_out_dict, nn_info_filepath)



