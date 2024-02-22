# flan_t5_qna_llamaindex.py
# Build a simple QnA bot with Flan T5 and llamaindex using a given set 
# of documents to reference.
# Windows/MacOS/Linux
# Python 3.10


import os
# from langchain.document_loaders import TextLoader				# load text files
# from langchain.text_splitter import CharacterTextSplitter		# text splitter
# from langchain.embeddings import HuggingFaceEmbeddings			# to use HuggingFace models
# from langchain.vectorstores import FAISS						# vector DB/store
# from langchain.chains.question_answering import load_qa_chain
# from langchain import HuggingFaceHub
# from langchain.llms import HuggingFaceHub
# from langchain.document_loaders import UnstructuredPDFLoader	# load pdf files
# from langchain.indexes import VectorstoreIndexCreator			# vectorize db with ChromaDB
# from langchain.chains import RetrievalQA						# combines a Retriever with QnA chain to do question answering
# from langchain.document_loaders import UnstructuredURLLoader	# load urls into document loader


# from langchain.document_loaders import PyPDFLoader						# load pdf files
# from langchain.text_splitter import RecursiveCharacterTextSplitter		# text splitter
# from langchain.embeddings import HuggingFaceEmbeddings					# to use HuggingFace models
# from langchain.vectorstores import FAISS								# vector DB/store
# # from langchain import HuggingFaceHub									# get (llm) model from huggingface hub
# from langchain.llms import HuggingFaceHub									# get (llm) model from huggingface hub
# from langchain.chains.question_answering import load_qa_chain			# loads a chain that you can use to do QA over a set of documents, but it uses ALL of those documents
# from langchain.chains import RetrievalQA								# combines a Retriever with QnA chain to do question answering


# from llama_index.core import ServiceContext
# # from llama_index import LLMPredictor
# # from llama_index import OpenAIEmbedding
# from llama_index.embeddings.huggingface import HuggingfaceEmbedding			# to use HuggingFace models
# from llama_index import PromptHelper
# # from llama_index.llms import OpenAI									#
# from llama_index.llms import huggingface							# get (llm) model from huggingface hub
# from llama_index.text_splitter import TokenTextSplitter				# text splitter
# from llama_index.node_parser import SimpleNodeParser				#
# from llama_index import VectorStoreIndex, SimpleDirectoryReader		# vector DB/store and data loader (respectively)
# from llama_index import set_global_service_context					#


from llama_index.core import VectorStoreIndex
from llama_index.core import SimpleDirectoryReader
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def main():
	# API token.
	with open('.env') as f:
		os.environ["HUGGINGFACEHUB_API_TOKEN"] = f.read()

	###################################################################
	# Load data
	###################################################################
	# As we know, LLMs do not possess updated knowledge of the world
	# nor knowledge about your internal documents. To help LLMs, we
	# need to feed them with relevant information from knowledge
	# sources. These knowledge sources can be structured data such as
	# CSV, Spreadsheets, or SQL tables, unstructured data such as
	# texts, Word Docs, Google Docs, PDFs, or PPTs, and semi-structured
	# data such as Notion, Slack, Salesforce, etc.
	# This articlewill use PDFs. Llama Index includes a class
	# SimpleDirectoryReader, which can read saved documents from a
	# specified directory. It automatically selects a parser based on
	# file extension.
	# You can have your custom implementation of a PDF reader using
	# packages like PyMuPDF or PyPDF2.
	documents = SimpleDirectoryReader(
		input_dir='../../data'
	).load_data()

	###################################################################
	# Chunk data
	###################################################################

	text_splitter = TokenTextSplitter(
		separator=" ",
		chunk_size=1024,
		chunk_overlap=20,
		backup_separators=["\n"],
		tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
	)
	node_parser = SimpleNodeParser.from_defaults(
		text_splitter = TokenTextSplitter()
	)


	###################################################################
	# Embed data
	###################################################################


	###################################################################
	# Store embeddings
	###################################################################


	###################################################################
	# Search embeddings
	###################################################################


	###################################################################
	# Querying
	###################################################################


	###################################################################
	# Afterward/Conclusion
	###################################################################


	# Exit the program.
	exit(0)


if __name__ == '__main__':
	main()