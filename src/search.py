import os
from dotenv import load_dotenv
from langchain_postgres import PGVector
from src.utils import get_connection_string, check_env_vars, get_embeddings_model, v_print


class DocumentSearcher:
    """
    Uma classe para encapsular a lógica de busca de documentos em um vector store.
    """
    def __init__(self, provider: str, collection_name: str = "documentos_pdf", verbose: bool = False):
        """
        Inicializa o buscador, configurando embeddings e a conexão com o banco.
        """
        self.verbose_print = v_print(verbose)
        self.verbose_print(f"Inicializando o buscador de documentos com o provedor: {provider}...")
        check_env_vars(provider)
        
        self.embeddings = get_embeddings_model(provider, verbose)
        self.connection_string = get_connection_string()
        self.collection_name = collection_name

        try:
            self.db = PGVector(
                embeddings=self.embeddings,
                collection_name=self.collection_name,
                connection=self.connection_string,
            )
            self.verbose_print("Conexão com o banco de dados vetorial estabelecida com sucesso.")
        except Exception as e:
            raise ConnectionError(f"Não foi possível conectar ao banco de dados: {e}") from e

    def search_documents(self, query: str, k: int = 10):
        """
        Realiza uma busca por similaridade no banco de vetores.
        """
        self.verbose_print(f"Buscando por: '{query}'...")
        similar_docs = self.db.similarity_search_with_score(query, k=k)
        self.verbose_print(f"Encontrados {len(similar_docs)} documentos similares.")
        return similar_docs

# Carrega as variáveis de ambiente do arquivo .env no escopo global
load_dotenv()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Busca documentos em um banco de dados vetorial.")
    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=['google', 'openai'],
        help="O provedor de LLM a ser usado: 'google' ou 'openai' (padrão: google)."
    )
    parser.add_argument(
        "--query", 
        type=str, 
        default="qualquer coisa", 
        help="A query para a busca (padrão: 'qualquer coisa')."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Aumenta a verbosidade para exibir logs detalhados."
    )
    args = parser.parse_args()

    # Este teste assume que o 'ingest' foi executado com o provedor 'google'
    print(f"--- Teste da classe de busca (provedor: {args.provider}) ---")
    try:
        searcher = DocumentSearcher(provider=args.provider, verbose=args.verbose)
        results = searcher.search_documents(args.query)
        
        if results:
            for doc, score in results:
                print("-" * 50)
                print(f"Score: {score}")
                print(f"Conteúdo: {doc.page_content[:200]}...")
                print(f"Metadados: {doc.metadata}")
                print("-" * 50)
        else:
            print("Nenhum resultado encontrado.")
            
    except (EnvironmentError, ConnectionError) as e:
        print(f"Erro: {e}")
    except Exception as e:
        print(f"\nOcorreu um erro inesperado durante o teste: {e}")