####
# Notebook 01
####
using CSV
using DataFrames
using CategoricalArrays
using Statistics

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
    "ibq_soot_t1"
]

inputs_de_interesse = select(raw_data, vcat(input_cols, [ "fe_varinha_tl_t4" ]))

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

## Definir a arquitetura/hiperparametros do modelo
rf_model = MLJDecisionTreeInterface.RandomForestRegressor()
rf_machine = machine(rf_model, 🙏[: , input_cols], 🙏[: , "fe_varinha_tl_t4"], cache=false)
MLJ.fit!(rf_machine, verbosity=0)

## 4. Construir o modelo



## 5. Treinar o modelo



## 6. Avaliar (validar) o modelo

previsoes = MLJ.predict(rf_machine, 🙏[: , input_cols]) 

using CairoMakie
fig = Figure(; size = (800,600))
ax = Axis(fig[1,1])
scatter!(ax, 🙏[: , "fe_varinha_tl_t4"], previsoes)
save("scatter.png", fig)
