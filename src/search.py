import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

def get_connection_string() -> str:
    """Constrói a string de conexão a partir das variáveis de ambiente."""
    user = os.getenv("POSTGRES_USER")
    password = os.getenv("POSTGRES_PASSWORD")
    host = os.getenv("POSTGRES_HOST")
    port = os.getenv("POSTGRES_PORT")
    db = os.getenv("POSTGRES_DB")
    return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"

def check_env_vars(provider: str):
    """Verifica se as variáveis de ambiente necessárias estão definidas."""
    db_vars = ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB"]
    if not all(os.getenv(var) for var in db_vars):
        missing = [var for var in db_vars if not os.getenv(var)]
        raise EnvironmentError(f"Variáveis de banco de dados ausentes: {', '.join(missing)}")

    if provider == 'google' and not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("Para o provedor 'google', a GOOGLE_API_KEY é necessária.")
    elif provider == 'openai' and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Para o provedor 'openai', a OPENAI_API_KEY é necessária.")

def get_embeddings_model(provider: str, verbose: bool = False):
    """Retorna a instância do modelo de embeddings com base no provedor."""
    verbose_print = v_print(verbose)
    if provider == 'google':
        verbose_print("Usando o modelo de embeddings do Google (models/embedding-001).")
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    elif provider == 'openai':
        verbose_print("Usando o modelo de embeddings da OpenAI (text-embedding-ada-002).")
        return OpenAIEmbeddings()
    else:
        raise ValueError("Provedor inválido. Escolha 'google' ou 'openai'.")

def v_print(verbose: bool):
    """Retorna uma função de print que só imprime se verbose for True."""
    def print_if_verbose(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    return print_if_verbose


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