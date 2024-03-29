# Flan T5 RAG with Langchain

Description: This is a quick example of using the Flan T5 model Langchain for retrieval augmented generation. 


### Setup

 - It is recommended that you create a python virtual environment instead of a conda due to version issues with a lot of necessary packages.
 - To set up the virtual environment, install the `venv` package:
     - `pip3 install virtualenv`
 - Create the new virtual environment:
     - `python -m venv pt-hf-onnx`
 - Activate the virtual environment:
     - Linux/MacOS: `source pt-hf-onnx/bin/activate`
     - Windows: `.\pt-hf-onnx\Scripts\activate`
 - Deactivate the virtual environment:
     - `deactivate`
 - Install the necessary packages (while the virtual environment is active):
     - `(pt-hf-onnx) pip3 install -r requirements.txt`
 - Also be sure to install the necessary version of `pytorch` according to your OS (refer to the `pytorch` website but the following command will help):
     - Linux & Windows (CUDA 11.8): `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
 - Regarding installing `faiss`, there are a few caveats with the library to consider:
     - `faiss` runs only on linux based operating systems (Linux and MacOS). To operate the module on windows, either use Docker or a WSL Linux distro (ie ubuntu)
     - `faiss` has GPU support only for NVIDIA devices. If you are installing the module on a Mac device, install the cpu-only version (`faiss-cpu`) when using either a MacOS or a machine without an NVIDIA GPU


### Notes

     - Upon running the conversion script provided by Xenova's `transformer.js` repo (command was `python transformers.js/scripts/convert.py --quantize --model_id google/flan-t5-small`), the program would return the following error: `TypeError: quantize_dynamic() got an unexpected keyword argument 'optimize_model'`
         - After looking at the code itself, there were no quantize parameters for the Flan-T5 model, causing the script to fail the way it did
         - Removing the `--quantize` argument yielded almost the same output as using the `optimum-cli` command above, with the exception that Xenova's script would organize the `.onnx` model files into their own folder `onnx/` while the `optimum-cli` command would put all outputs in one select folder. The results will yield the same error when trying to use `transformers.js` as above
 - Flan-T5 model sizes (according to [this medium article](https://medium.com/@koki_noda/try-language-models-with-python-google-ais-flan-t5-ba72318d3be6)):
     - Flan T5 small - 80M (297MB on disk)
     - Flan T5 base - 250M (948MB on disk)
     - Flan T5 large - 780M (2.9GB on disk)
     - Flan T5 xl - 3B (11GB on disk)
     - Flan T5 xxl - 11B
 - Major benefit of Flan-T5 is that most of the model variants (with the exception of xl and/or xxl) can run relatively easily on consumer hardware (even an Apple Silicon macbook). The other LLMs (Llama 1/2, Falcon, Vicuna, Mistral, etc) need to be quantized because they're so large (smallest variant of each is about 7B parameters)
     - Other models that have models with smaller than 7B parameters include the following:
         - GPT 2 (124M, 335M, 774M, 1.5B)
         - GPT Neo (125M, 1.3B, 2.7B)
         - GPT J (6B)
         - OPT (125M, 350M, 1.3B, 2.7B, 6.7B, 13B, 30B, 66B, 175B)
         - OpenLlama (3B, 7B, 13B)
         - StableLM (3B, 7B)
     - Note that the models over 1B parameters may need to be quantized to run on consumer hardware. As seen in the QLora-Falcon7B and GPTQ-Falcon7B examples, quantization requires (usually an NVIDIA) GPU at the cost of some degredation in the quality of the output
 - Notes from `flan_t5_raw_inference.py`:
     - Flan-T5 works best when prompted when doing ConditionalGeneration. For instance: "what does George Bush do?" will return "president of united states" or "continue the text: the quick brown fox" gives "was able to run away from the tiger.". As you can see, the responses are quite short. It's not *ideal* for a ChatGPT alternative but seems quite capable.
     - Passing custom parameters to the model.generate() function yields much more detailed output. The downside is that it requires tuning and these parameters cannot be adjusted "on the fly" in a live application unless using a jupyter notebook.
         - Depending on the parameters 
     - Using AutoTokenizer and AutoModelForSeq2SeqLM give no warning messages when initializing the model compared to using the T5Tokenizer and T5ModelForConditionalGeneration classes.
 - Notes from `flan_t5_qna_langchain.py`:
     - Changing the model for the llm & embeddings affects the output of the RetrievalQA chain. No model in particular has proven to be the best so far. Additional tinkering (with both models and the llm model kwargs) is advised.
 - Notes from `flan_t5_transformers_js_inference/index.js`:
     - It is imperative that the Flan-T model be exported with Xenova's `transformers.js` conversion script under the package versions outlined in their `requirements.txt`. Failure to do so will lead to unsuccessful conversions or the converted model not working with the `transformers.js` library.
         - For more information on the conversion process, as well as links to the converted model, see [here](https://github.com/dmmagdal/HuggingfaceOfflineDownloader/tree/main/onnx_converter/README.md).
     - If a model was converted without the `--quantize` flag being set, then `{quantize: false}` needs to be passed into the appropriate `AutoModel` class (in this case, `AutoModelForSeq2Seq`) or `pipeline` class constructors. `transformers.js` will not look for quantized copies of the `.onnx` model files (see [here](https://github.com/xenova/transformers.js/blob/main/src/models.js#L123) in `PretrainedMixin.constructSession()`).


### References

 - transformers
     - [Huggingface T5 Documentation](https://huggingface.co/docs/transformers/model_doc/t5)
     - [Huggingface Flan T5 Documentation](https://huggingface.co/docs/transformers/model_doc/flan-t5)
     - [Huggingface Text Generation Config](https://huggingface.co/docs/transformers/v4.35.2/en/main_classes/text_generation#transformers.GenerationConfig) (used in `AutoModelForSeq2SeqLM.generate()`)
     - [TransformersJS Models Documentation](https://huggingface.co/docs/transformers.js/api/models): Models
     - [TransformersJS Pipelines Documentation](https://huggingface.co/docs/transformers.js/pipelines): Pipelines
     - [TransformersJS Pipelines Documentation](https://huggingface.co/docs/transformers.js/api/pipelines#module_pipelines.Text2TextGenerationPipeline): Text2TextGeneration Pipeline
     - [TransformersJS Guide](https://huggingface.co/docs/transformers.js/tutorials/node): Server-side Inference in Node.js
     - [TransformersJS Guide](https://huggingface.co/docs/transformers.js/custom_usage#convert-your-models-to-onnx): Export a model to ONNX
     - [Transformers Guide](https://huggingface.co/docs/transformers/serialization): Export to ONNX
     - [Optimum Guide](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model): Export a model to ONNX with optimum.exporters.onnx (optimum-cli)
     - [Huggingface Blog](https://huggingface.co/blog/convert-transformers-to-onnx): Convert Transformers to ONNX with Hugging Face Optimum
     - [huggingface Documentation](https://huggingface.co/docs/hub/repositories-getting-started): Getting Started with Repositories
     - [Huggingface Blog](https://huggingface.co/blog/getting-started-with-embeddings): Getting Started With Embeddings
 - tutorial
     - [Medium Article](https://blog.searce.com/building-a-video-assistant-leveraging-large-language-models-2e964e4eefa1): Building a video assistant leveraging Large Language Models
     - [LinkedIn Post](https://www.linkedin.com/pulse/small-overview-demo-o-google-flan-t5-model-balayogi-g/): A Small Overview and Demo of Google Flan-T5 Model
     - [Medium Article](https://betterprogramming.pub/is-google-flan-t5-better-than-openai-gpt-3-187fdaccf3a6): Is Google’s Flan-T5 Better Than OpenAI GPT-3?
     - [Analytics Vidhya Post](https://www.analyticsvidhya.com/blog/2023/09/unlocking-langchain-flan-t5-xxl-a-guide-to-efficient-document-querying/): Unlocking LangChain & Flan-T5 XXL | A Guide to Efficient Document Querying (disable javascript on the web page & reload it to bypass the sign in prompt)
     - [Analytics Vidhya Post](https://www.analyticsvidhya.com/blog/2023/07/building-llm-powered-applications-with-langchain/): Building LLM-Powered Applications with LangChain (disable javascript on the web page & reload it to bypass the sign in prompt)
     - [Medium Article](https://medium.com/@xiaohan_63326/fine-tune-fine-tuning-t5-for-text-generation-c51ed54a7941): [Fine Tune] Fine Tuning T5 for Text Generation
     - [Medium Article](https://medium.com/nlplanet/a-full-guide-to-finetuning-t5-for-text2text-and-building-a-demo-with-streamlit-c72009631887): A Full Guide to Finetuning T5 for Text2Text and Building a Demo with Streamlit
     - [YouTube Video](https://www.youtube.com/watch?v=_Qf_SiCLzw4&ab_channel=code_your_own_AI): NEW Flan-T5 Language model | CODE example | Better than ChatGPT?
     - [Mediucm Article](https://betterprogramming.pub/harnessing-retrieval-augmented-generation-with-langchain-2eae65926e82): Harnessing Retrieval Augmented Generation With Langchain
     - [YouTube Video](https://www.youtube.com/watch?v=mAYS4d0hrek&ab_channel=CodingCrashcourses): LangChain with JavaScript 🦜️ - COMPLETE TUTORIAL - Basics to advanced concept!
     - [Medium Article](https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed): Using langchain for Question Answering on Own Data
 - model
     - [flan-t5-small](https://huggingface.co/google/flan-t5-small)
     - [flan-t5-base](https://huggingface.co/google/flan-t5-base)
     - [flan-t5-large](https://huggingface.co/google/flan-t5-large)
     - [flan-t5-xl](https://huggingface.co/google/flan-t5-xl)
     - [flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl)
     - [Xenova/flan-t5-small](https://huggingface.co/Xenova/flan-t5-small) (transformers.js)
     - [Xenova/flan-t5-base](https://huggingface.co/Xenova/flan-t5-base) (transformers.js)
 - langchain
     - [Langchain Blog](https://blog.langchain.dev/): Home
     - [Langchain Blog](https://blog.langchain.dev/js-envs/): Announcing LangChainJS Support for Multiple JS Environments
     - Python
        - [Langchain Documentation](https://python.langchain.com/docs/get_started/introduction/): Introduction
        - [Langchain Documentation](https://python.langchain.com/docs/modules/model_io/llms/): LLMs 
        - [Langchain Documentation](https://python.langchain.com/docs/integrations/llms/huggingface_hub): LLMs > Huggingface Hub
        - [Langchain Documentation](https://python.langchain.com/docs/integrations/llms/huggingface_pipelines): LLMs > Huggingface Local Pipelines
        - [Langchain Documentation](https://python.langchain.com/docs/expression_language/cookbook/retrieval): Cookbook > RAG 
        - [Langchain Documentation](https://python.langchain.com/docs/modules/agents/): Agents 
     - JS
        - [LangchainJS Documentation](https://js.langchain.com/docs/get_started/introduction/): Introduction 
        - [LangchainJS Documentation](https://js.langchain.com/docs/integrations/llms/): LLMs
        - [LangchainJS Documentation](https://js.langchain.com/docs/integrations/llms/ollama): LLMs > Ollama 
        - [LangchainJS Documentation](https://js.langchain.com/docs/integrations/llms/huggingface_inference): LLMs > Huggingface Inference 
        - [LangchainJS Documentation](https://js.langchain.com/docs/expression_language/cookbook/retrieval): Cookcbook > RAG
        - [LangchainJS Documentation](https://js.langchain.com/docs/expression_language/cookbook/agents): Cookcbook > Agents
 - additional references
     - [Medium Article](https://betterprogramming.pub/is-google-flan-t5-better-than-openai-gpt-3-187fdaccf3a6): Is Google’s Flan-T5 Better Than OpenAI GPT-3?
     - [Exemplary AI Post](https://exemplary.ai/blog/flan-t5): What is FLAN-T5? Is FLAN-T5 a better alternative to GPT-3?
     - [Medium Article](https://medium.com/@koki_noda/try-language-models-with-python-google-ais-flan-t5-ba72318d3be6): Try Language Models with Python: Google AI’s Flan-T5 (premium article)
     - [Narrativa Post](https://www.narrativa.com/flan-t5-a-yummy-model-superior-to-gpt-3/): FLAN-T5, a yummy model superior to GPT-3
     - [Medium Article](https://medium.com/google-cloud/finetuning-flan-t5-base-and-online-deployment-in-vertex-ai-bf099c3a4a86): Fine-tuning Flan-T5 Base and online deployment in Vertex AI
     - [Huggingface Forum](https://discuss.huggingface.co/t/onnx-flan-t5-model-oom-on-gpu/36342): ONNX Flan-T5 Model OOM on GPU
     - [Huggingface Forum](https://discuss.huggingface.co/t/unable-to-import-faiss/3439): Unable to import faiss
     - [Microsoft Open Source Blog](https://cloudblogs.microsoft.com/opensource/2023/10/04/accelerating-over-130000-hugging-face-models-with-onnx-runtime/): Accelerating over 130,000 Hugging Face models with ONNX Runtime
