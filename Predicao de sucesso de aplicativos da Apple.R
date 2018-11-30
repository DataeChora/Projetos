library(dplyr)
library(GGally)
library(ggplot2)
library(NbClust)
library(caret)
library(randomForest)
library(ggdendro)
library(rattle)
library(rpart)
library(caretEnsemble)
library(pROC)
library(ggfortify)

### ADENDO: NÃO ESTÁ COMENTADO AINDA, WORKING ON IT...

### LENDO OS DADOS ######################################################

apple <- read.csv(file="AppleStore.csv", header = T, sep = ",")

variaveis_interessantes <- dput(c("size_bytes","price","user_rating",
                                  "user_rating_ver","cont_rating",
                                  "prime_genre","rating_count_ver",
                                  "sup_devices.num","lang.num","rating_count_tot",
                                  "ipadSc_urls.num"))

### ANALISE EXPLORATORIA ###############################################

head(apple)
names(apple)
ggpairs(apple[-c(1,2,3,5,11,13)])
View(apple)


apple1 <- apple[-c(1,2,3,5,11,12,13)] %>% 
  mutate_if(is.integer, as.numeric)

summary(apple)
str(apple1)
apple1 <- apple[, -c(1,2,3,5,11,12)]
names(apple1)
apple.pca1 <- prcomp(apple1[,-c(7,11)], center=TRUE, scale.=TRUE)
autoplot(apple.pca, data=apple1)
summary(apple.pca1)
apple.pca$eig

##  Podemos notar a natureza assimetrica dos dados, o que futuramente poderia ser um empecilho na analise,
## verificamos tambem, que pelas estatisticas sumarias, algumas variaveis corroboram a informacao anterior,
## tais variaveis como "price", "rating_count_tot" e "size_bytes" sao assimetricos e interessantes o sufici-
## entes para uma futura analise.

##### CLUSTERIZAÇÃO HIERARQUICA #################################

app1 <- apple %>%
  select(variaveis_interessantes) %>%
  group_by(prime_genre) %>%
  summarise_all(mean)

app1 <- data.frame(app1)
rownames(app1) <- levels(apple$prime_genre)
app_dist <- dist(scale(app1[,-c(1,6)]), method = "manhattan")
ggdendrogram(hclust(app_dist))

## OBS: Uma clusterizacao foi feita para separarmos em grupos a variável prime_genre, 
## sendo utilizada a média das variáveis númericas com a finalidade de categoriza-las.

### TRATANDO OS DADOS ###################################################
head(app)
app <- apple %>%
  select(variaveis_interessantes) %>%
  mutate(size_bytes = log(size_bytes),
         user_ratingn = (rating_count_tot*user_rating - rating_count_ver*user_rating_ver)/(rating_count_tot-rating_count_ver),
         price = ifelse(apple$price == 0, "Gratis", ifelse(apple$price <= 7.475, "Barato", ifelse(apple$price > 7.475,"Caro","NA"))),
         user_rating_ver = ifelse(apple$user_rating_ver < 4.5, "regular","bom"),
         lang.num = log(lang.num+1),
         rating_count_totn = log(rating_count_tot-rating_count_ver+1))%>%
  filter(rating_count_tot > 1) %>% mutate_if(is.character, as.factor)
app = na.omit(app)
levels(app$prime_genre) <- c("1","3","1","4","2","1","1","4","2","1","4","3","1","1","2","3","4","2","2","1","2","2","2")
levels(app$prime_genre) <- c("Grupo_1","Grupo_3","Grupo_4","Grupo_2")
app$prime_genre <- as.factor(app$prime_genre)
app = app %>%
  select(-c(rating_count_tot,rating_count_ver,user_rating))

#### CART ###########################################################
set.seed(572613)
index <- createDataPartition(app$user_rating_ver,
                             p=0.75, list=FALSE)
app_treino <- app[ index, ]
app_teste <- app[-index, ]

fitControl <- trainControl(method = "cv",
                           number = 5,
                           savePred = TRUE,
                           classProb = TRUE)

tune.grid <- expand.grid(cp = seq(-0.5, 0.5, 0.05))

app_cart <- train(user_rating_ver ~ .,
                  data = app_treino,
                  method = "rpart",
                  tuneGrid = tune.grid,
                  trControl = fitControl)

ggplot(app_cart)

confusionMatrix(app_cart)
head(app)
predicao <- predict(app_cart, app_teste)
confusionMatrix(predicao, app_teste$user_rating_ver)

ggplot(varImp(app_cart))

fancyRpartPlot(app_cart$finalModel)

#### COMBINAÇAO DE MODELOS #############################################
set.seed(572613)
index <- createDataPartition(app$user_rating_ver, p = .8, list = F)
app_treino <- app[index,]
app_teste <- app[-index,]

fitControl <- trainControl(method = "cv", number = 5,
                           classProbs = T, summaryFunction = twoClassSummary)
knn_config <- caretModelSpec(method = "knn",
                             tuneGrid = expand.grid(k = seq(3,19,2)))
rf_config <- caretModelSpec(method = "rf",
                            tuneGrid = expand.grid(mtry=1:9))
nnet_config <- caretModelSpec(method = "nnet",
                              tuneGrid = expand.grid(size = seq(1,10,1), decay = seq(0.1,0.5,0.1)))

lista_modelos <- caretList(
  user_rating_ver ~ ., data=app_treino,
  trControl=fitControl,
  metric="Accuracy",
  tuneList=list(
    knn=knn_config,
    rf=rf_config,
    nnet=nnet_config)
)

combinacao_1 <- caretEnsemble(
  lista_modelos,
  metric="Accuracy",
  trControl=fitControl)

predicao_combinacao_1 <- predict(combinacao_1,app_teste,
                                 type="prob")
resultado_combinacao_1 <- roc(app_teste$user_rating_ver,
                              predicao_combinacao_1)
ggroc(resultado_combinacao_1)

summary(combinacao_1)

modelCor(resamples(lista_modelos))

ggplot(varImp(combinacao_1$ens_model))
ggplot(varImp(lista_modelos$knn))
ggplot(varImp(lista_modelos$rf))
ggplot(varImp(lista_modelos$nnet))

