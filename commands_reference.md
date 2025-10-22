conda activate ~/miniconda3/envs/py37

python -m zhiyuan.retriever.bm25anserini.evaluate_anserini_bm25 --dataset_name fiqa

python train_sbert.py --dataset_name fiqa --train_num 50 --exp_name no_aug --weak_num 100k

docker run -it --rm -e JAVA_OPTS="-Xmx8g" -p 8002:8000 beir/pyserini-fastapi


#for SPTAR_DPR_EVAL

 python zhiyuan/dpr_eval.py --dataset_name fiqa --version v1 --gpu_id 0 --train_num 50 --weak_num 100k --exp_names llama_7b_100k_fixed_v3_best_llama_prompt_2_filtered_70

#for inpair DPR

python zhiyuan/dpr_eval.py --dataset_name fiqa --version v1 --gpu_id 0 --train_num 50 --weak_num 100k --exp_names p_written_100k_vicuna_prompt_2_filtered_70
