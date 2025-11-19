import funcao

def main():
    arquivo = 'disney_plus_titles.csv'
    print("O Grafo esta sendo criado")
    grafo = funcao.cria_grafo(arquivo)
    print("Grafo criado com sucesso")

    while True:
        titulo_filme = input("\n(digite 'sair' para finalizar) Digite o nome do filme: ")
        if titulo_filme.lower() == 'sair':
            break
        funcao.recomenda(titulo_filme,grafo)
        

if __name__ == "__main__":
    main() 