#Script para chamar todos os experimentos e guarda-los em uma pasta especifica
#exigindo a saida de um arquivo com nome experimentos e o comando 7z instalado
numeroExecucoes=1
pastaDestino='experimentoGrupoDeRnaDiferente1_2_3_4_5_6_7_8_9_10_11_12'
comando="nice -n 20 python facaMeuTrabalhoTodosMesesComRnasAgrupadas.py"
#
mkdir $pastaDestino
for (( i=0; i < $numeroExecucoes; i++ ))
do
    $comando
    mv experimentos $pastaDestino/experimentos_$i
done
echo terminando, vou comprimir
7z a $pastaDestino.7z $pastaDestino
echo "tudo pronto :-))"
