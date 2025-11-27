import os
import argparse
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from src.search import DocumentSearcher, check_env_vars

def get_chat_model(provider: str, verbose: bool = False):
    """Retorna a instância do modelo de chat com base no provedor."""
    verbose_print = v_print(verbose)
    if provider == 'google':
        verbose_print("Usando o modelo de chat do Google (gemini-2.5-flash).")
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    elif provider == 'openai':
        verbose_print("Usando o modelo de chat da OpenAI (gpt-3.5-turbo).")
        return ChatOpenAI(temperature=0) # Modelo padrão é gpt-3.5-turbo
    else:
        raise ValueError("Provedor inválido. Escolha 'google' ou 'openai'.")

def v_print(verbose: bool):
    """Retorna uma função de print que só imprime se verbose for True."""
    def print_if_verbose(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    return print_if_verbose

def format_context(docs_with_scores, verbose_print):
    """Formata os documentos recuperados em uma string de contexto."""
    context = []
    verbose_print("\n--- Documentos Recuperados ---")
    for i, (doc, score) in enumerate(docs_with_scores):
        source_info = f"Fonte: {os.path.basename(doc.metadata.get('source', 'N/A'))}, Página: {doc.metadata.get('page', 'N/A')}"
        context.append(f"{doc.page_content}\n({source_info})")
        verbose_print(f"Doc {i+1} (Score: {score:.4f}): {source_info}\n{doc.page_content[:100]}...")
    verbose_print("--------------------------\n")
    return "\n\n---\n\n".join(context)

def main():
    """
    Função principal para iniciar o chat CLI.
    """
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Inicia um chat com um documento PDF.")
    parser.add_argument(
        "--provider",
        type=str,
        default="google",
        choices=['google', 'openai'],
        help="O provedor de LLM a ser usado: 'google' ou 'openai' (padrão: google)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Aumenta a verbosidade para exibir logs detalhados e fontes de contexto."
    )
    args = parser.parse_args()
    verbose_print = v_print(args.verbose)

    try:
        check_env_vars(args.provider)
    except EnvironmentError as e:
        print(f"Erro de configuração: {e}")
        return

    try:
        searcher = DocumentSearcher(provider=args.provider, verbose=args.verbose)
        llm = get_chat_model(args.provider, args.verbose)
    except (ConnectionError, ValueError) as e:
        print(f"Erro na inicialização: {e}")
        return

    prompt_template = """CONTEXTO:
{context}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

PERGUNTA DO USUÁRIO:
{question}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm | StrOutputParser()

    print(f"--- Chat com Documento PDF (Provedor: {args.provider}) ---")
    print("Digite sua pergunta ou 'sair' para terminar.")

    while True:
        try:
            question = input("\nPergunta: ")
            if question.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando o chat. Até logo!")
                break
            if not question.strip():
                continue

            relevant_docs = searcher.search_documents(question)
            context = format_context(relevant_docs, verbose_print) if relevant_docs else ""
            
            if not context:
                verbose_print("Nenhum documento relevante encontrado para a consulta.")

            verbose_print("\nGerando resposta...")
            response = chain.invoke({"context": context, "question": question})
            print("\nResposta:")
            print(response)

        except Exception as e:
            print(f"\nOcorreu um erro durante o chat: {e}")

if __name__ == '__main__':
    main()
