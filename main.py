
import funcao

def main():
    arquivo = 'disney_plus_titles.csv'
    print("O Grafo esta sendo criado")
    grafo = funcao.cria_grafo(arquivo)
    print("Grafo criado com sucesso")

    while True:
        titulo_usuario = input("\n(digite 'sair' para finalizar) Digite o nome do filme: ")
        if titulo_usuario.lower() == 'sair':
            break
        funcao.recomenda(titulo_usuario,grafo)
        

if __name__ == "__main__":
    main() 