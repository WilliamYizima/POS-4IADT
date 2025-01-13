###############################################################################
# Projeto TCC1 - Modelos possíveis para supervisionado e não supervisionado
###############################################################################

# Eeftua a carga das bibliotecas necessárias
# Este modelo é uma exceção e utiliza bibliotecas diversas

# Imports padrões
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import keyboard

# SKLearn..
from sklearn import datasets

# Metricas
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC  # Maquina de Vetor Suporte SVM
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier  # RandomForest
from sklearn.neighbors import KNeighborsClassifier  # k-vizinhos mais próximos (KNN)

# Cross-Validation models.
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz

import warnings

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

#------------------------------------------
#carrega o dataset original, sem tratamento
#insurance_dataset = pd.read_csv('G:\\Users\\ricar\\Downloads\\insuranceV3.csv',decimal=',',sep=';')
insurance_dataset = pd.read_csv('G:\\Users\\ricar\\Downloads\\TCC_1 - IA - FIAP\\insurance Original 2.csv', decimal=',', sep=';')

# Conjunto de comandos para avaliar o dataset
#mostra os cabeçalhos
print(insurance_dataset.head(10))
#keyboard.read_key()

#mostra quantidade de regs x cols
print(insurance_dataset.shape)
#keyboard.read_key()

# infos gerais
print(insurance_dataset.info())
#keyboard.read_key()

#verifica se existem nulls
print(insurance_dataset.isnull().sum())
#keyboard.read_key()

#apresenta a descricao dos dados do dataset
print(insurance_dataset.describe())
#keyboard.read_key()

#apresenta linhas duplicadas
print(insurance_dataset.duplicated())
#keyboard.read_key()

# Apresenta todas as colunas
print(insurance_dataset.columns)
#keyboard.read_key()

# Apresenta o somatorio e um grafico de cada coluna
for coluna in insurance_dataset.columns :
    insurance_dataset.value_counts(coluna)

#keyboard.read_key()

plt.figure(figsize=(6,6))

# Apresentações gráficas e avaliações
sns.regplot(data=insurance_dataset, x="encargos", y="idade", logx=True)
plt.show()

# distribuição da idade
#sns.set()
sns.histplot(insurance_dataset['idade'])
plt.title('Distribuição por idade')
plt.show()

# coluna sexo
sns.countplot(x='sexo', data=insurance_dataset)
plt.title('Distribuição por sexo')
plt.show()

# total de homens e mulheres
insurance_dataset['sexo'].value_counts()

# distribuição por imc
#Normal IMC na faixa de → 18.5 até 24.9
sns.histplot(insurance_dataset['imc'])
plt.title('Distribuição por IMC')
plt.show()

# filhos
sns.countplot(x='filhos', data=insurance_dataset)
plt.title('Distribuição por filhos')
plt.show()

# totais de registros por qtd de filhos
insurance_dataset['filhos'].value_counts()

# fumante..
sns.countplot(x='fumante', data=insurance_dataset)
plt.title('Distribuição por fumante')
plt.show()

# soma os fumantes e não
insurance_dataset['fumante'].value_counts()

# região
sns.countplot(x='regiao', data=insurance_dataset)
plt.title('Distribuição por região')
plt.show()

# total por região
insurance_dataset['regiao'].value_counts()

# Valores de encargos
sns.histplot(insurance_dataset['encargos'])
plt.title('Distribuição por encargos')
plt.show()

# Valores gerais
plt.figure(figsize=(16,16))
sns.pointplot(insurance_dataset)
plt.title('Distribuição geral')
plt.show()

# Antes de montar o mapa de correlação, gerar um dataset sem literais.
# Uma primeira conversão sem limpeza..
# nova var
ins_sem_literal = insurance_dataset.copy()
# Encode SEXO, FUMANTE e REGIÃO
le = LabelEncoder()
for col in ins_sem_literal.columns:
    if ins_sem_literal[col].dtype == 'object':
        ins_sem_literal[col] = le.fit_transform(ins_sem_literal[col])

# Mapa para correlação entre colunas
sns.heatmap(ins_sem_literal.corr(), annot=True, cmap='Greens')
plt.title('Correlação geral')
plt.show()

# Precisa remover idades = -3, 999, 499, 500, 111 porque é fumante e 2 porque possui 4 filhos.
# Existem linhas com IMC nulo. Como IMC é importante, não compensa colocar valores médios.
# Pode deturpar os resultados. Também serão removidas.
# O dataset limpo é gravado como insurance_fase1 que possui os literais e ins_sem_literal já transformado.
insurance_fase1 = ins_sem_literal.copy()
# Separar em um vetor as idades que desejamos remover
# Removendo as linhas com as idades especificadas
idades_del = [-3, 2, 103, 111, 499, 500, 999]
# A propriedade IS IN pega todos os registros de idade iguais a lista
# O ~ é negação. Inverte todos os booleanos do anterior, ficando TRUE os que não são as idades.
# Finalmente leva para o dataset somente os index true.
insurance_fase1 = insurance_fase1[~insurance_fase1['idade'].isin(idades_del)]
# Remove todas as linhas que contenham algum campo nulo. No caso, só na coluna IMC tem nulos.
insurance_fase1.dropna(axis=0, how='any', inplace= True)
# Apresenta o total de linhas agora
insurance_fase1.shape
# Apresenta as primeiras 5 linhas apos a limpeza
print(insurance_fase1.head())

# Reapresenta os graficos comparativos apos limpeza
# Criacao de uma figura e dois subplots lado a lado
# Criando a figura com dois subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plotando o primeiro gráfico
sns.regplot(data=insurance_dataset, x="encargos", y="idade", logx=True, ax=ax1)
ax1.set_title("Relação entre Encargos e Idade - Conjunto de Dados 1")

# Plotando o segundo gráfico
sns.regplot(data=insurance_fase1, x="encargos", y="idade", logx=True, ax=ax2)
ax2.set_title("Relação entre Encargos e Idade - Conjunto de Dados 2")

plt.tight_layout()
plt.show()

# distribuição da idade
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plotando o primeiro gráfico
sns.histplot(insurance_dataset['idade'], ax=ax1)
ax1.set_title("Distribuição por idade - Conjunto de Dados 1")

# Plotando o segundo gráfico
sns.histplot(insurance_fase1['idade'], ax=ax2)
ax2.set_title("Distribuição por idade - Conjunto de Dados 2")

plt.tight_layout()
plt.show()

# distribuição por imc
#Normal IMC na faixa de → 18.5 até 24.9
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plotando o primeiro gráfico
sns.histplot(insurance_dataset['imc'], ax=ax1)
ax1.set_title("Distribuição por IMC - Conjunto de Dados 1")

# Plotando o segundo gráfico
sns.histplot(insurance_fase1['imc'], ax=ax2)
ax2.set_title("Distribuição por IMC - Conjunto de Dados 2")

plt.tight_layout()
plt.show()

# Mapa para correlação entre colunas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

# Plotando o primeiro gráfico
sns.heatmap(ins_sem_literal.corr(), annot=True, cmap='Greens', ax=ax1)
ax1.set_title("Correlação geral - Conjunto de Dados 1")

# Plotando o segundo gráfico
sns.heatmap(insurance_fase1.corr(), annot=True, cmap='Greens', ax=ax2)
ax2.set_title("Correlação geral - Conjunto de Dados 2")

plt.tight_layout()
plt.show()

# Aqui começamos a preparar para regressão
# Equação de regressão linear simples
# Yi = I + a * Xi + Ei
# diz que x é tudo menos encargos e y são os encargos
X = insurance_fase1.drop(columns='encargos', axis=1)
Y = insurance_fase1['encargos']
print(X)
print(Y)

#separa para treino 80% e teste 20%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Carrega o modelo de regressão
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None)

# treina o dado para ser preditivo
training_data_previsao = regressor.predict(X_train)
# trabalha com quadrados para eliminar negativos
r2_train = metrics.r2_score(Y_train, training_data_previsao)
print('R**2 treino : ', r2_train)

# efetivando os testes
test_data_previsao =regressor.predict(X_test)
# Idem
r2_test = metrics.r2_score(Y_test, test_data_previsao)
print('R**2 teste : ', r2_test)

# idade, sexo, imc, filhos, fumante, regiao, amigos, inimigos, chips, fritas, miojo, comportamento
input_vetor = [(31, 1, 25.74, 0, 1, 0, 7, 0, 1, 1, 1, 1),
               (29, 1, 35.00, 2, 0, 2, 2, 7, 6, 6, 3, 7),
               (45, 0, 44.74, 0, 1, 2, 5, 5, 5, 5, 5, 5),
               (18, 0, 53.04, 0, 0, 3, 6, 2, 3, 2, 3, 6),
               (55, 0, 30.43, 0, 0, 4, 10, 0, 1, 1, 1, 1)]

# loop com 5 testes e resultados
for lista in input_vetor:
    input_data = lista
    # converte input_data para um array numpy
    input_data_as_numpy_array = np.asarray(input_data)

    # Redefine o vetor
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Faz a previsão
    previsao = regressor.predict(input_data_reshaped)
    print(previsao)

    # Mostra as previsões
    print('Os encargos para uma pessoa com ',input_data[0], ' anos, do sexo ', input_data[1],' com massa corporal de ',input_data[2],' com ',input_data[3], ' filhos, fumante ',input_data[4], ' morador na região ', input_data[5], ' em USD', previsao)

keyboard.read_key()

# Aqui termina a regresão 1
#-------------------------------
# Inicio da regressão multipla
df = insurance_fase1.copy()

# Adicionar uma constante para o termo de intercepto
df['Intercepto'] = 1

# Definir as variáveis independentes (X)
X = df[['Intercepto', 'idade', 'imc', 'sexo', 'filhos', 'fumante', 'regiao', 'amigos', 'inimigos', 'chips', 'fritas', 'miojo', 'comportamento']]

# Definir a variável dependente (Y)
Y = df['encargos']

# Criar e ajustar o modelo de regressão linear múltipla
modelo = sm.OLS(Y, X).fit()

# Imprimir os resultados do modelo
print(modelo.summary())

#---------------------------------

# realizar as previsões
y_pred = modelo.predict(X)

# Calcular MAE, MSE e RMSE
mae = mean_absolute_error(Y, y_pred)
mse = mean_squared_error(Y, y_pred)
rmse = np.sqrt(mse)

# Imprimir as métricas
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

# Aqui termina a regressão multipla
#----------------------------------------------------
# Aqui inicia o PCA - Preparando para o pre-processamento.
# Busca os melhores parametros
df = insurance_fase1.copy()

# A seguir, separaremos todas as colunas na lista de ‘recursos’ para uma variável ‘X’ e a variável ‘destino’ para ‘y’.
features = ['idade', 'sexo', 'imc', 'fumante', 'filhos', 'regiao', 'amigos', 'inimigos', 'chips', 'fritas', 'miojo', 'comportamento']
X = df[['idade', 'sexo', 'imc', 'fumante', 'filhos', 'regiao', 'amigos', 'inimigos', 'chips', 'fritas', 'miojo', 'comportamento']] .values
y = df['encargos'].values

# Normalizando os dados utilizando o standardScaler
# (Padroniza as features removendo a média e escala a variância a uma unidade.
# Isso significa que para cada feature, a média seria 0, e o Desvio Padrão seria 1)
X = StandardScaler().fit_transform(X)
#Visualizando nossos dados padronizados
df_padronizado = pd.DataFrame(data=X, columns=features)
df_padronizado.head()

# Instanciando o pca e a quantidade de componentes que desejamos obter
pca = PCA(n_components=4)
# Aplicando PCA nas nossas features
principalComponents = pca.fit_transform(X)

# Criando um novo dataframe para visualizarmos como ficou nossos dados reduzidos com o PCA
df_pca = pd.DataFrame(data = principalComponents,
                  columns = ['PC1', 'PC2', 'PC3', 'PC4'])
print(df_pca)

target = pd.Series(insurance_fase1['encargos'], name='encargos')
result_df = pd.concat([df_pca, target], axis=1)
result_df

print('Variance of each component:', pca.explained_variance_ratio_)
print('Total Variance Explained:', round(sum(list(pca.explained_variance_ratio_))*100, 2))

# Aqui termina o PCA
#-------------------------------------------
# Aqui inicia
np = insurance_fase1.copy()
# Criando a variável alvo (1 para aprovado, 0 para não aprovado)
# Os valores são aleatorios.. para testar..
alvo = np.where((idade < 64) & (encargos  > 100), 1, 0)

# Criando o DataFrame
data = insurance_fase1.copy()

# Embaralhando o conjunto de dados
data_shuffled = data.sample(frac=1, random_state=42)

# Salvando o conjunto de dados embaralhado em um arquivo CSV - para exemplo
data_shuffled.to_csv('dataset_embaralhado_com_features.csv', index=False)

# Carregando o conjunto de dados fictício de previsão de custo
data = data_shuffled

# Pré-processamento dos dados
# Vamos supor que o conjunto de dados possui as seguintes colunas: 'idade', 'encargos', 'fumante', 'alvo' (0 para não aprovado, 1 para aprovado)
# Separando features (X) e target (y)
X = data.drop('alvo', axis=1)
y = data['alvo']

# Dividindo o conjunto de dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar o scaler usando apenas o conjunto de treino
scaler = StandardScaler()
scaler.fit(X_train)
# Aplicar o Z-score nas features de treino
X_train_scaled = scaler.transform(X_train)
# Aplicar o Z-score nas features de teste usando as estatísticas do conjunto de treino
X_test_scaled = scaler.transform(X_test)

# Aqui termina de deixar padronizado para o scaler
#------------------------------------------------------
# Aqui começa o KNN

modelo_classificador = KNeighborsClassifier(n_neighbors=5)
modelo_classificador.fit(X_train, y_train)

error = []
for i in range(1, 10): #range de tentativas para k
    knn = KNeighborsClassifier(n_neighbors=i)# aqui definimos  o k
    knn.fit(X_train, y_train) #treinando para encontrar o erro
    pred_i = knn.predict(X_test_scaled) #armazenando as previsões
    error.append(np.mean(pred_i != y_test)) #armazenando o valor do erro médio na lista de erros

# Aqui o SVC
svm = Pipeline([("linear_svc", LinearSVC(C=1))])
svm.fit(X_train, y_train)
# SVM
poly_svm = Pipeline([("svm", SVC(kernel="poly", degree=3, coef0=1, C=5))])
svm.fit(X_train, y_train)
# KMeans
kmeans = KMeans(n_clusters=5, random_state=7)
kmeans.fit(data)

# Armazena o SSE (soma dos erros quadraticos) para cada quantidade de k
sse = []

# Roda o K-means para cada k fornecido
for i in k:
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(data[['idade','encargos','fumante','sexo']])
    sse.append(kmeans.inertia_) #calculo do erro do k-mens (mudar o centroide dos dados)

    plt.rcParams['figure.figsize'] = (10, 5)
    # Plota o gráfico com a soma dos erros quadraticos
    plt.plot(k, sse, '-o')
    plt.xlabel(r'Número de clusters')
    plt.ylabel('Inércia')
    plt.show()

#arvores
#subindo a base de dados
dados = insurance_fase1.coy()

#Separando os dados em treino e teste:
x = dados.drop(columns=['encargos'])
y = dados['encargos'] #O que eu quero prever. (Target)

#Criando o modelo de árvore de decisão:
dt = DecisionTreeClassifier(random_state=7, criterion='entropy', max_depth = 2)
dt.fit(X_train, y_train)
dt.predict(X_test)

#Plotando a árvore: Neste caso precisa ajustar a arvore
tree.plot_tree(dt)
class_names = ['Fraude', 'Não Fraude']
label_names = ['distance_from_home', 'distance_from_last_transaction',	'ratio_to_median_purchase_price',	'repeat_retailer',	'used_chip',	'used_pin_number',	'online_order']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=300)

tree.plot_tree(dt, feature_names = label_names, class_names=class_names, filled = True)
fig.savefig('imagename.png')

# floresta randomica
rf = RandomForestClassifier(n_estimators=5, max_depth = 2,  random_state=7)

rf.fit(X_train, y_train)

y_predito_random_forest = rf.predict(X_test)

fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
for index in range(0, 5):
    tree.plot_tree(rf.estimators_[index], feature_names = label_names, class_names=class_names, filled = True, ax = axes[index])
    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)

fig.savefig('rf_5trees.png')

# Validação Cruzada
kfold  = KFold(n_splits=5, shuffle=True)
result = cross_val_score(modelo_classificador, x, y, cv = kfold)
print("K-Fold (R^2) Scores: {0}".format(result))
print("Mean R^2 for Cross-Validation K-Fold: {0}".format(result.mean()))

def AplicaValidacaoCruzada(x_axis, y_axis):
# Configuração de KFold.
    kfold = KFold(n_splits=10, shuffle=True)
    # Axis
    x = x_axis
    y = y_axis

    # Criando os modelos
    # KNN
    knn = KNeighborsClassifier(n_neighbors=9, metric= 'cosine', weights='distance')
    knn.fit(X_train_scaled, y_train)

    # SVM
    svm = SVC()
    svm.fit(X_train_scaled, y_train)

    # RandomForest
    rf = RandomForestClassifier(random_state=7)
    rf.fit(X_train_scaled, y_train)

    # Applyes KFold to models.
    knn_result = cross_val_score(knn, x, y, cv = kfold)
    svm_result = cross_val_score(svm, x, y, cv = kfold)
    rf_result = cross_val_score(rf, x, y, cv = kfold)

    # Creates a dictionary to store Linear Models.
    dic_models = {
        "KNN": knn_result.mean(),
        "SVM": svm_result.mean(),
        "RF": rf_result.mean()
    }
    # Select the best model.
    melhorModelo = max(dic_models, key=dic_models.get)

    print("KNN (R^2): {0} SVM (R^2): {1} Random Forest (R^2): {2}".format(knn_result.mean(), svm_result.mean(), rf_result.mean()))
    print("O melhor modelo é : {0} com o valor: {1}".format(melhorModelo, dic_models[melhorModelo]))

# Modelando hiperparametros
# Parâmetros testados
param_grid = {'n_neighbors': [9, 14], 'weights': ['uniform', 'distance'], 'metric': ['cosine', 'euclidean', 'manhattan']}
gs_metric = make_scorer(accuracy_score, greater_is_better=True)

grid = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid, scoring=gs_metric, cv=5, n_jobs=4, verbose=3)
grid.fit(X_train_scaled, y_train)
knn_params = grid.best_params_
print('KNN', knn_params)
#Fitting 5 folds for each of 12 candidates, totalling 60 fits
#KNN {'metric': 'cosine', 'n_neighbors': 9, 'weights': 'distance'}

grid.cv_results_

# Matriz de confusão
matriz_confusao = confusion_matrix(y_true = y_test,
                                   y_pred = y_predito_random_forest,
                                   labels=['Disk Hernia', 'Normal', 'Spondylolisthesis'])
# plotando uma figura com a matriz de confusao
figure = plt.figure(figsize=(15, 5))

disp = ConfusionMatrixDisplay(confusion_matrix = matriz_confusao,
                              display_labels=['Disk Hernia', 'Normal', 'Spondylolisthesis'])
disp.plot(values_format='d')

#from sklearn.metrics import classification_report
print(classification_report(y_test, y_predito_random_forest))

# Curva ROC e AUC
y_prob = modelo_classificador.predict_proba(X_test_scaled)[:,1]
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc

plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

