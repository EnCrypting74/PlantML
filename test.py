def tuning_KNN_custom():
    # Suddividi il dataset in training set e test set
    data = 'Total'
    train_x, test_x, train_y, test_y = DS_Splitter(data)

    # Definisci i range di valori per k e tipi di distanza
    k_range = list(range(1, 32))
    dist_range = ["Euclidean", "Manhattan", "Chebyshev"]
    model_stats = {}

    # Inizializza array per memorizzare le accuratezze durante il tunin

    # Loop attraverso le diverse distanze e valori di k
    for dist in dist_range:
        for k in k_range:
            # Crea un classificatore KNN personalizzato con la configurazione corrente
            clf = Custom_kNN(k = k, distance = dist)
            print("Distanza: ",dist, "  K: ", k)

            # Calcola l'accuratezza media sul set di test
            predictions = clf.fit_predict(train_x,train_y,test_x)

            acc, pre , rec , f1 = tuningMetrics(predictions, test_y)

            model_stats[f'{dist} with {k} neighbours'] = [acc, pre, rec, f1]
    
    best_model = max(model_stats, key=model_stats.get)
    max_values = model_stats[best_model]

    return ("metriche migliori = ",best_model, " con ", max_values)

def tuning_RF_custom():
    data = 'Total'
    train_x, test_x, train_y, test_y = DS_Splitter(data)

    # Definisci i range di valori per k e tipi di distanza
    trees_range = list(range(5, 100,10))
    depth_range = list(range(10,50,5))
    model_stats = {}
    
    for trees in trees_range:
        for depth in depth_range:
            RF = CustomRandomForest(trees,depth)
            print("Alberi :",trees, " Profondità : ",depth)
            predictions = RF.fit_predict(train_x,train_y,test_x)

            acc, pre , rec , f1 = tuningMetrics(predictions, test_y)

            model_stats[f'{trees} trees with depth : {depth}'] = [acc, pre, rec, f1]
    
    best_model = max(model_stats, key=model_stats.get)
    max_values = model_stats[best_model]

    return ("metriche migliori = ",best_model, " con ", max_values)

print(tuning_RF_custom())