---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:24894
- loss:RLFeedbackLoss
base_model: google-bert/bert-base-uncased
widget:
- source_sentence: Is alcoholism a bad thing
  sentences:
  - ' Why would somebody want an IRA if they have a 401K and a Roth 401K?'
  - ' Drinking is debatable, alcoholism is net negative, drunk driving is net negative.
    Alcohol on its own doesn''t kill. It''s not quite the same as cigarettes. Maybe
    we just start with products that kill, that way the moral question isn''t so blurry?'
  - ' Financial advisors are a client facing role and their utilization of math is
    relatively limited as far as I am aware. Most of the bigger PWM/AWM groups do
    the analytical work at a head office and the FA''s in the field are basically
    account men. Their entire livelihood is based around relationship management with
    their clients.'
- source_sentence: Robinhood is not a good investment tool
  sentences:
  - ' &gt;Many studies have shown that index funds and passive investing are the most
    successful strategies for users. This is the opposite of what Robinhood encourages.  I''ve
    been with Robinhood awhile now... and I have only used it to purchase shares of
    index ETFs.  I have never felt encouraged by them to day trade or actively manage
    investments.'
  - ' The Romanian mountains are very steep, with extremely different geography, offering
    great scenes and genuine high encounters.  mountain biking in Transylvania Romania
    We''ve been directing individuals coming here from everywhere throughout the world,
    be it for climbing, trekking, biking, and more. We have confidence in a reasonable
    business approach. Our visitors ought to have an awesome time while mountain biking
    in Romania, our convenience, transportation suppliers ought to be upbeat, and
    we ought to be happy with our work and advantages.'
  - ' how is it double taxation when you didn''t start off with that extra $100?  it''s
    double taxation if they taxed you on the total amount you pulled out of the market,
    not the profit you made.   explain the math on your last part, please.'
- source_sentence: What's the difference between anarchy and a democracy
  sentences:
  - ' On Friday morning, Amazon shares open with a bounce of 8 percent against Thursday’s
    closing. This increase added $ 7 billion in the net worth of the company’s head
    Jeff Bezos. But on the other hand, Microsoft’s shares increased by just 7 percent,
    due to which Jeff became the richest person in the world once again, surpassing
    Bill Gates.Registration Begins Nov 7  current business news - wikifeed  According
    to a Forbes report, the increase of 2 percent in Ajman’s shares increased the
    total assets of Bezos by $ 90 million and became the world’s richest man. Their
    assets increased to $ 90.6 billion, which is slightly higher than Bill Gates’s
    assets ($ 90.1 billion)'
  - ' &gt; You know what scares the shit out of giant militaries?  And again you avoid
    my question. Instead of giving a reason why your proposal won''t devolve into
    anarchy you just claim my example of anarchy is a good thing. I''ll bet in school
    when asked a math question about a guy buying a dozen cantaloupes at the store
    you just claimed that nobody needed that many cantaloupes and expected full credit.  I''m
    now convinced you haven''t seriously considered the effects of what you propose.'
  - ' Few extra dollars? Sometimes $10+ a ticket, which quickly becomes $20 when taking
    someone out. As someone who goes to a few shows a week (so far two this week,
    third this Saturday), that would be $60 a week, but sadly I do not have a girlfriend.   I
    actually didn''t think I would have an example from this week, but look at [Laura
    Marling''s](http://www.ticketmaster.com/event/0700475301BFB681?artistid=1362337&amp;majorcatid=10001&amp;minorcatid=1)
    show in Chicago. The **service fees are $14.34**. I actually saw her at a venue
    this week primarily run by volunteers (also not ticketmaster) and the show was
    a quarter the cost. When service fees cost as much (sometimes more) as a concert
    there is something wrong.'
- source_sentence: Apple's Scott Forstall is leaving the company
  sentences:
  - ' I remember reading that [he was responsible for siri](http://www.businessinsider.com/scott-forstall-apple-maps-2012-9)
    and that launch also didn''t go so well.  I guess Apple figured that 0 for 2 is
    better than 0 for 3 so better to get rid of his ass now before he fucks up again.
    And from other reports, people are cheering his departure ( [it''s said that Forstall''s
    coworkers were so excited to show him the door that they volunteered to split
    up his workload](http://www.theverge.com/2012/10/29/3574022/apple-scott-forstall-ios-6-maps-apology-letter))
    so don''t see how this is a bad move, other than Google or Samsung might pick
    him up, which I guess might be bad news for the apple crowd.'
  - ' Lol was scrolling through and reread it and decided to address the other part
    of your comment :) he knows a friend working at his dads company making that much
    a year and they are in dire need of more welders lol so its like nearly guaranteed
    since he is a close friend and the dudes dad pays them all the same and doesnt
    give his son xtra lol so solid 80k.'
  - ' Speeding and distracted driving are among the most common reasons for getting
    a traffic ticket. Consult with Stephen G. Price, a seasoned defence lawyer in
    Langley, if you need help in defending your rights and preventing your license
    from being revoked.'
- source_sentence: How does a company issue corporate bonds
  sentences:
  - ' To issue corporate grade bonds the approval process very nearly matches that
    for issuing corporate equity. You must register with the sec, and then generally
    there is a initial debt offering similar to an IPO. (I say similar in terms of
    the process itself, but the actual sale of bonds is nothing like that for equities).
    It would be rare for a partnership to be that large as to issue debt in the form
    of bonds (although there are some that are pretty big), but I suppose it is possible
    as long as they want to file with the sec.   Beyond that a business could privately
    place bonds with a large investor but there is still registration requirements
    with the sec.  All that being said, it is also pretty rare for public bonds to
    be issued by a company that doesn''t already have public equity. And the amounts
    we are talking about here are huge. The most common trade in corporate debt is
    a round lot of 100,000. So this isn''t something a small corporation would have
    access to or have a need for. Generally financing for a smaller business comes
    from a bank.'
  - ' "What does your comment have to do with my comment? You say ""Apple only designs
    stuff"" as if that has some bearing on their net worth. Right now Apple''s stock
    is worth about as much as *Google and Microsoft combined*, and they''re sitting
    on about 60B in *cash*. They are *extremely* wealthy. They could do absolutely
    nothing for a very very long time and still stay in business."'
  - ' Hola, I am a nontraditional student, meaning I am in my 30''s. I was laid off
    in 2009 because of the economy and was encouraged by the unemployment agency to
    goto school AND get student loans. Now I am completeley screwed!!! Are any of
    you out there screwed because of student loans? AMA!'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy@1
- cosine_accuracy@3
- cosine_accuracy@5
- cosine_accuracy@10
- cosine_precision@1
- cosine_precision@3
- cosine_precision@5
- cosine_precision@10
- cosine_recall@1
- cosine_recall@3
- cosine_recall@5
- cosine_recall@10
- cosine_ndcg@10
- cosine_mrr@10
- cosine_map@100
model-index:
- name: SentenceTransformer based on google-bert/bert-base-uncased
  results:
  - task:
      type: information-retrieval
      name: Information Retrieval
    dataset:
      name: dev
      type: dev
    metrics:
    - type: cosine_accuracy@1
      value: 0.238
      name: Cosine Accuracy@1
    - type: cosine_accuracy@3
      value: 0.35
      name: Cosine Accuracy@3
    - type: cosine_accuracy@5
      value: 0.41
      name: Cosine Accuracy@5
    - type: cosine_accuracy@10
      value: 0.486
      name: Cosine Accuracy@10
    - type: cosine_precision@1
      value: 0.238
      name: Cosine Precision@1
    - type: cosine_precision@3
      value: 0.1413333333333333
      name: Cosine Precision@3
    - type: cosine_precision@5
      value: 0.1056
      name: Cosine Precision@5
    - type: cosine_precision@10
      value: 0.0674
      name: Cosine Precision@10
    - type: cosine_recall@1
      value: 0.12334769119769118
      name: Cosine Recall@1
    - type: cosine_recall@3
      value: 0.20290252525252522
      name: Cosine Recall@3
    - type: cosine_recall@5
      value: 0.2489271284271284
      name: Cosine Recall@5
    - type: cosine_recall@10
      value: 0.30451370851370846
      name: Cosine Recall@10
    - type: cosine_ndcg@10
      value: 0.24848717387807948
      name: Cosine Ndcg@10
    - type: cosine_mrr@10
      value: 0.31062539682539675
      name: Cosine Mrr@10
    - type: cosine_map@100
      value: 0.20429265696363938
      name: Cosine Map@100
---

# SentenceTransformer based on google-bert/bert-base-uncased

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) <!-- at revision 86b5e0934494bd15c9632b12f734a8a67f723594 -->
- **Maximum Sequence Length:** 350 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 350, 'do_lower_case': False, 'architecture': 'BertModel'})
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'How does a company issue corporate bonds',
    " To issue corporate grade bonds the approval process very nearly matches that for issuing corporate equity. You must register with the sec, and then generally there is a initial debt offering similar to an IPO. (I say similar in terms of the process itself, but the actual sale of bonds is nothing like that for equities). It would be rare for a partnership to be that large as to issue debt in the form of bonds (although there are some that are pretty big), but I suppose it is possible as long as they want to file with the sec.   Beyond that a business could privately place bonds with a large investor but there is still registration requirements with the sec.  All that being said, it is also pretty rare for public bonds to be issued by a company that doesn't already have public equity. And the amounts we are talking about here are huge. The most common trade in corporate debt is a round lot of 100,000. So this isn't something a small corporation would have access to or have a need for. Generally financing for a smaller business comes from a bank.",
    " Hola, I am a nontraditional student, meaning I am in my 30's. I was laid off in 2009 because of the economy and was encouraged by the unemployment agency to goto school AND get student loans. Now I am completeley screwed!!! Are any of you out there screwed because of student loans? AMA!",
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[ 1.0000,  0.7158, -0.0936],
#         [ 0.7158,  1.0000, -0.0562],
#         [-0.0936, -0.0562,  1.0000]])
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Information Retrieval

* Dataset: `dev`
* Evaluated with [<code>InformationRetrievalEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.InformationRetrievalEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| cosine_accuracy@1   | 0.238      |
| cosine_accuracy@3   | 0.35       |
| cosine_accuracy@5   | 0.41       |
| cosine_accuracy@10  | 0.486      |
| cosine_precision@1  | 0.238      |
| cosine_precision@3  | 0.1413     |
| cosine_precision@5  | 0.1056     |
| cosine_precision@10 | 0.0674     |
| cosine_recall@1     | 0.1233     |
| cosine_recall@3     | 0.2029     |
| cosine_recall@5     | 0.2489     |
| cosine_recall@10    | 0.3045     |
| **cosine_ndcg@10**  | **0.2485** |
| cosine_mrr@10       | 0.3106     |
| cosine_map@100      | 0.2043     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 24,894 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                        | sentence_1                                                                           | label                        |
  |:--------|:----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:-----------------------------|
  | type    | string                                                                            | string                                                                               | int                          |
  | details | <ul><li>min: 3 tokens</li><li>mean: 11.72 tokens</li><li>max: 50 tokens</li></ul> | <ul><li>min: 13 tokens</li><li>mean: 138.95 tokens</li><li>max: 350 tokens</li></ul> | <ul><li>1: 100.00%</li></ul> |
* Samples:
  | sentence_0                                             | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                             | label          |
  |:-------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------|
  | <code>How much does a good acoustic guitar cost</code> | <code> Now you can get a variety of musical instrument for sale by reaching out to an experienced music store. Such shops offer everything from guitars to resonators, dulcimers, ukuleles and mandolin for people who are into acoustic music.</code>                                                                                                                                                                 | <code>1</code> |
  | <code>Why are cities so expensive</code>               | <code> "Because cities are expensive as fuck.  I always here my peers complaining about not being able to afford housing and whatnot.   Here I am, 23, and the owner of a nice little 3 bedroom house. Just move somewhere smaller.  My city is about 40k pop. And it's only a 45 minute drive to the center of the closest major city, and there are plenty of ""suburbs"" even closer with good job markets."</code> | <code>1</code> |
  | <code>Alcatel 2021I Dual Stage Rotary Vane Pump</code> | <code> Fully refurbished to factory standards. The Alcatel 2021I Dual Stage Rotary Vane Pump is coupled with a NW25 Inlet Flange and NW25 Outlet Flange. This Alcatel Rotary Vane Pump has a peak pumping velocity of 14.6 CFM and a final pressure of 1 x 10-3 Torr. It has been thoroughly examined and is ready for broad range of applications within the research and industrial industries.</code>               | <code>1</code> |
* Loss: <code>__main__.RLFeedbackLoss</code>

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `num_train_epochs`: 2
- `fp16`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: no
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 32
- `per_device_eval_batch_size`: 32
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 2
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `project`: huggingface
- `trackio_space_id`: trackio
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: no
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: True
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step | Training Loss | dev_cosine_ndcg@10 |
|:------:|:----:|:-------------:|:------------------:|
| 0.6427 | 500  | -0.3374       | -                  |
| 1.0    | 778  | -             | 0.2350             |
| 1.2853 | 1000 | -0.6708       | -                  |
| 1.9280 | 1500 | -0.7239       | -                  |
| 2.0    | 1556 | -             | 0.2485             |


### Framework Versions
- Python: 3.13.7
- Sentence Transformers: 5.1.1
- Transformers: 4.57.0
- PyTorch: 2.8.0+cu128
- Accelerate: 1.10.1
- Datasets: 3.3.2
- Tokenizers: 0.22.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->