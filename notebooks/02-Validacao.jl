####
# Notebook 01
####
using CSV
using DataFrames
using CategoricalArrays
using Statistics
using Random

## 1. Obter dados

# Essa linha vai ler o CSV dos dados que a gente tem (banco completo)
raw_data = CSV.read("C:\\Users\\Guilherme\\Documents\\WorkshopBrainrise2023\\GerminaModelingWorkshop\\ext\\merged_data.csv", DataFrame)

input_cols = readlines("C:\\Users\\Guilherme\\Documents\\WorkshopBrainrise2023\\GerminaModelingWorkshop\\ext\\variaveis_input_inicial.txt")

output_cols = [
    "fe_cat_rev_1_t4",
    "fe_cat_rev_2_t4",
    "fe_gire_mt_t4",
    "fe_gire_cor_t4",
    "fe_gire_per_t4",
    "fe_varinha_tl_t4",
    "vin_com_padrao_t1",
    "vin_soc_padrao_t1",
    "ibq_reg_t1",
    "ibq_dur_t1",
    "ibq_soot_t1",
    "ibq_reg_t2",
    "ibq_dur_t2",
    "ibq_soot_t2",
    "bayley_3_t3"
]

output_col = "bayley_3_t3"

inputs_de_interesse = select(raw_data, vcat(input_cols, [ output_col ]))

## 2. Processar os dados

# Verificar a quantidade de missings
missingcols = sort(DataFrame(
    :variavel => names(inputs_de_interesse),
    :n_missings => map(x -> sum(ismissing.(x)), eachcol(inputs_de_interesse))
), :n_missings)

🙏 = dropmissing(inputs_de_interesse)

function z_normalizar(vv)
    this_mean = Statistics.mean(vv)
    this_sd = Statistics.std(vv)
    return( (vv .- this_mean) ./ this_sd )
end

🙏 = transform!(🙏, names(🙏) .=> ( x -> z_normalizar(x)) .=> names(🙏) ; renamecols = false)

## 3. Decidir o modelo

# Depois de decidir o modelo,

## 3.1. Localizar as bibliotecas que FAZEM esse modelo na sua linguagem
using MLJ
using MLJDecisionTreeInterface
using DecisionTree

## 3.2 Carregar o modelo
RandomForestRegressor = MLJ.@load RandomForestRegressor pkg=DecisionTree

## 4. Construir o modelo
## Definir a arquitetura/hiperparametros do modelo
rf_model = MLJDecisionTreeInterface.RandomForestRegressor()
rf_machine = machine(rf_model, 🙏[: , input_cols], 🙏[: , output_col], cache=false)

## 5. Treinar o modelo
MLJ.fit!(rf_machine, verbosity=0)

## 6. Avaliar (validar) o modelo

## Primeiro, vamos ver se o modelo funciona para o conjunto de TREINO
# Porque se aqui ele ja falhar, nao tem esperanca nenhuma pra ele.
previsoes = MLJ.predict(rf_machine, 🙏[: , input_cols]) 

# Plotando
using CairoMakie
fig = Figure(; size = (800,600))
ax = Axis(fig[1,1])
scatter!(ax, 🙏[: , output_col], previsoes)
save("scatter.png", fig)

# Olhando a importancia das variaveis
importances = DataFrame(
    :variable => names(🙏[: , input_cols]),
    :importance => impurity_importance(rf_machine.fitresult)
)

sort!(importances, :importance; rev = true)[1:15, :]

## Agora sim vamos validar de fato o modelo

# emparalhar as linhas da matriz de dados
Random.seed!(0)
🙏 = 🙏[Random.randperm(nrow(🙏)), :]

X_treino = 🙏[1:30, input_cols]
y_treino = 🙏[1:30, output_col]

X_teste = 🙏[31:45, input_cols]
y_teste = 🙏[31:45, output_col]

fig = Figure(; size = (800,600))
ax1 = Axis(fig[1,1])
ax2 = Axis(fig[1,2])
xlims!(ax1, (-1,+5))
xlims!(ax2, (-1,+5))
hist!(ax1, y_treino, color = :blue)
hist!(ax2, y_teste, color = :orange)
save("hist_sets.png", fig)

# refazer o treinamento, so com os dados de treino agora
rf_model = MLJDecisionTreeInterface.RandomForestRegressor()
rf_machine = machine(rf_model, X_treino, y_treino, cache=false)
MLJ.fit!(rf_machine, verbosity=0)

# gerar as previsoes

previsoes_treino = MLJ.predict(rf_machine, X_treino) #yhat_train
previsoes_teste = MLJ.predict(rf_machine, X_teste) #yhat_test

# Figuras de Merito
# Modelo de REGRESSAO:
## - MAE (Mean Absolute Error) (L1 loss) = sum(abs( yhat - y )) / n_dados
MAE_treino = sum(abs.(previsoes_treino .- y_treino)) / length(y_treino)
MAE_teste = sum(abs.(previsoes_teste .- y_teste)) / length(y_teste)
## MSE (Mean SQUARE Erroor) (L2 loss) 
MSE_treino = sum((previsoes_treino .- y_treino).^2) / length(y_treino)
MSE_teste = sum((previsoes_teste .- y_teste).^2) / length(y_teste)
## Correlacao de Pearson 
R_treino = Statistics.cor(previsoes_treino, y_treino)
R_teste = Statistics.cor(previsoes_teste, y_teste)
## Scatteplot
# Plotando
using CairoMakie

fig = Figure(; size = (800,600))
ax = Axis(fig[1,1])
xlims!(ax, (-1,+5))
ylims!(ax, (-1,+5))
scatter!(ax, y_treino, previsoes_treino, color = :blue, size = 5)
scatter!(ax, y_teste, previsoes_teste, color = :orange, size = 5)
ablines!(ax, [0], [1], color = :red)
save("scatter_val.png", fig)