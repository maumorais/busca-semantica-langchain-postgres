import os
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector
from src.utils import get_connection_string, check_env_vars, get_embeddings_model, v_print

def main():
    """
    Script principal para ingerir um documento PDF no banco de dados vetorial.
    """
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Ingere um documento PDF no banco de dados vetorial.")
    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=['google', 'openai'],
        help="O provedor de LLM a ser usado: 'google' ou 'openai' (padrão: google)."
    )
    parser.add_argument(
        "--path", 
        type=str, 
        default="document.pdf", 
        help="O caminho para o arquivo PDF a ser processado (padrão: document.pdf)."
    )
    parser.add_argument(
        "--collection", 
        type=str, 
        default="documentos_pdf", 
        help="O nome da coleção no banco de dados vetorial (padrão: documentos_pdf)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Aumenta a verbosidade para exibir logs detalhados."
    )
    args = parser.parse_args()
    verbose_print = v_print(args.verbose)

    try:
        check_env_vars(args.provider)
    except EnvironmentError as e:
        print(f"Erro de configuração: {e}")
        return

    verbose_print(f"Lendo o arquivo: {args.path}...")
    try:
        loader = PyPDFLoader(args.path)
        docs = loader.load()
        verbose_print(f"Documento carregado com {len(docs)} página(s).")
    except FileNotFoundError:
        print(f"Erro: O arquivo '{args.path}' não foi encontrado.")
        return
    except Exception as e:
        print(f"Ocorreu um erro ao ler o PDF: {e}")
        return

    verbose_print("Dividindo o documento em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    verbose_print(f"Documento dividido em {len(split_docs)} chunks.")

    try:
        embeddings = get_embeddings_model(args.provider, args.verbose)
    except ValueError as e:
        print(f"Erro: {e}")
        return

    connection_string = get_connection_string()
    collection_name = args.collection

    verbose_print(f"Salvando embeddings na coleção '{collection_name}' (Provedor: {args.provider})...")
    PGVector.from_documents(
        embedding=embeddings,
        documents=split_docs,
        collection_name=collection_name,
        connection=connection_string,
        pre_delete_collection=True,
    )

    print("\nProcesso de ingestão concluído com sucesso!")

if __name__ == '__main__':
    main()