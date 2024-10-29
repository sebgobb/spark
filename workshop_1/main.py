from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, GBTClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder.appName("Practice").getOrCreate()

# DATAFRAME RÉCUPÉRÉ D'APRÈS LE CSV MUSHROOMS.CSV
df_pyspark = spark.read.csv("mushrooms.csv", inferSchema=True, header=True)

# Impression du type de valeur de chaque colonne
df_pyspark.printSchema()

#########################################################################################

### I. POUR CHAQUE COLONNE AVEC DES VALEURS DE TYPE STRING, ENCODAGE EN TYPE INT

categorical_columns = [
        "class",
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "odor",
        "gill-attachment",
        "gill-spacing",
        "gill-size",
        "gill-color",
        "stalk-shape",
        "stalk-root",
        "stalk-surface-above-ring",
        "stalk-surface-below-ring",
        "stalk-color-above-ring",
        "stalk-color-below-ring",
        "veil-type",
        "veil-color",
        "ring-number",
        "ring-type",
        "spore-print-color",
        "population",
        "habitat"
    ]

# ENCODAGE
for category in categorical_columns:
    stringIndexer = StringIndexer(inputCol = category, outputCol = category+"_encoded").fit(df_pyspark)
    df_pyspark = stringIndexer.transform(df_pyspark)
    df_pyspark = df_pyspark.withColumn(category+"_encoded", df_pyspark[category+"_encoded"].cast('int'))

# COLONNES ENCODEES
categorical_columns_digitized = [
        "class_encoded",
        "cap-shape_encoded",
        "cap-surface_encoded",
        "cap-color_encoded",
        "bruises_encoded",
        "odor_encoded",
        "gill-attachment_encoded",
        "gill-spacing_encoded",
        "gill-size_encoded",
        "gill-color_encoded",
        "stalk-shape_encoded",
        "stalk-root_encoded",
        "stalk-surface-above-ring_encoded",
        "stalk-surface-below-ring_encoded",
        "stalk-color-above-ring_encoded",
        "stalk-color-below-ring_encoded",
        "veil-type_encoded",
        "veil-color_encoded",
        "ring-number_encoded",
        "ring-type_encoded",
        "spore-print-color_encoded",
        "population_encoded",
        "habitat_encoded"
    ]

encoded_df = df_pyspark.select(categorical_columns_digitized)

# VÉRIFICATION
encoded_df.show(2)

#########################################################################################

### II. EXTRACTION DES FEATURES

input_columns = [
        "cap-shape_encoded",
        "cap-surface_encoded",
        "cap-color_encoded",
        "bruises_encoded",
        "odor_encoded",
        "gill-attachment_encoded",
        "gill-spacing_encoded",
        "gill-size_encoded",
        "gill-color_encoded",
        "stalk-shape_encoded",
        "stalk-root_encoded",
        "stalk-surface-above-ring_encoded",
        "stalk-surface-below-ring_encoded",
        "stalk-color-above-ring_encoded",
        "stalk-color-below-ring_encoded",
        "veil-type_encoded",
        "veil-color_encoded",
        "ring-number_encoded",
        "ring-type_encoded",
        "spore-print-color_encoded",
        "population_encoded",
        "habitat_encoded"
]

featureAssemBCer = VectorAssembler(inputCols=input_columns,outputCol="features")

output = featureAssemBCer.transform(encoded_df)

print("\nTABLEAU DES FEATURES et CLASSES")
output.select("features","class_encoded").show(5, truncate=False)

#########################################################################################

# RANDOMSPLIT POUR LES DONNÉES D'ENTRAÎNEMENT ET LES DONNÉES DE TEST
train, test = output.randomSplit([0.8, 0.2], seed=17,)

print("----------------------------------------")
print("TAILLE DES DONNÉES D'ENTRAÎNEMENT :", train.count())
print("TAILLE DES DONNÉES DE TEST :", test.count())
print("----------------------------------------")

# Filtrage des lignes où "class_encoded" est égal à 1
filtered_test = test.filter(test["class_encoded"] == 1)

# Compter le nombre de lignes filtrées
count_ones = filtered_test.count()

print("\nNombre de fois où la valeur '1' apparaît dans 'class_encoded' :", count_ones)

#########################################################################################

### III. COMPARAISON DES MODÈLES D'IA

listAUC = {}

### III.1 RÉGRESSION LOGISTIQUE
print("\nClassification par RÉGRESSION LOGISTIQUE : ")

# Entraînement du modèle
rl = LogisticRegression(featuresCol = "features", labelCol = "class_encoded", maxIter=10)
rlModel = rl.fit(train)

# Prédiction
predictions = rlModel.transform(test)
predictions.select("class_encoded","features","rawPrediction","probability","prediction").show(5, truncate=False)

# Évaluation du modèle
evaluator = BinaryClassificationEvaluator(labelCol="class_encoded")
rlAUC = evaluator.evaluate(predictions)
listAUC["Régression logistique"] = rlAUC

print("\n\t*** AUC (Régression logistique) :", rlAUC)

#########################################################################################

### III.2 ARBRE DE DÉCISION
print("\nClassification par ARBRE DE DÉCISION : ")

# Entraînement du modèle
ad = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'class_encoded', maxDepth = 3)
adModel = ad.fit(train)

# Prédiction
predictions = adModel.transform(test)
predictions.select("class_encoded","features","rawPrediction","probability","prediction").show(5, truncate=False)

# Évaluation des performances
evaluator = BinaryClassificationEvaluator(labelCol="class_encoded")
adAUC = evaluator.evaluate(predictions)
listAUC["Arbre de décision"] = adAUC

print("\n\t*** AUC (Arbre de décision):", adAUC)

#########################################################################################

### III.3 RANDOM FOREST
print("\nClassification par RANDOM FOREST :")

# Entraînement du modèle
rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'class_encoded', numTrees = 500, maxDepth = 10)
rfModel = rf.fit(train)

# Prédiction
predictions = rfModel.transform(test)
predictions.select("class_encoded","features","rawPrediction","probability","prediction").show(5, truncate=False)

# Évaluation des performances
evaluator = BinaryClassificationEvaluator(labelCol="class_encoded")
rfAUC = evaluator.evaluate(predictions)
listAUC["Random forest"] = rfAUC

print("\n\t*** AUC (Random Forest) :", rfAUC)

#########################################################################################

### III.4 SUPPOR VECTOR MACHINE
print("\nClassification par SUPPORT VECTOR MACHINE :")

# Entraînement du modèle
svm = LinearSVC(featuresCol="features", labelCol="class_encoded", maxIter=10)
svmModel = svm.fit(train)

# Prédiction
predictions = svmModel.transform(test)
predictions.select("class_encoded","features","prediction").show(5, truncate=False)

# Évaluation des performances
evaluator = BinaryClassificationEvaluator(labelCol="class_encoded")
svmAUC = evaluator.evaluate(predictions)
listAUC["Support Vector Machine"] = svmAUC

print("\n\t*** AUC (Support Vector Machine) : ", svmAUC)

#########################################################################################

### III.5 GRADIENT BOOSTING MACHINES
print("\nClassification par GRADIENT BOOSTING MACHINES :")

# Entraînement du modèle
gbt = GBTClassifier(featuresCol="features", labelCol="class_encoded", maxIter=10)
gbtModel = gbt.fit(train)

# Prédiction
predictions = gbtModel.transform(test)
predictions.select("class_encoded","features","rawPrediction","probability","prediction").show(5, truncate=False)

# Évaluation des performances
evaluator = BinaryClassificationEvaluator(labelCol="class_encoded")
gbtAUC = evaluator.evaluate(predictions)
listAUC["Gradien Boosting Machines"] = gbtAUC

print("\n\t*** AUC (Gradient Boosting) : ", gbtAUC)

#########################################################################################

### VI. COMPARATIF
classement = sorted(listAUC.items(), key=lambda item: item[1], reverse=True)
print(f"\nClassement des modèles ayant les meilleures performances : ")
for i, (k, v) in enumerate(classement, start=1):
    print(f"\t{i}. {v:.3f} ---> {k} ")
