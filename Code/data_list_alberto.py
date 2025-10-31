#!/usr/bin/env python

# %%
# Aedes aegypti: 1390 (612) 2 AREE
# Aedes albopictus: 1189 (200)
# Aedes caspius: 1024 (320)
# Aedes koreicus: 936 (676)
# Anopheles maculipennis: 1094 (711)
# Culex pipiens: 1114 (160)

# Culex theileri: 632 (632) SINGOLA AREA
# Aedes vexans: 443 (160)
# Aedes geniculatus: 366 (79)
# Aedes mariae: 260 (70)
# Culex hortensis: 247 (50)
# Culiseta longiareolata: 172 (35)
# Anopheles algeriensis: 171  (171) SINGOLA AREA
# Aedes rusticus: 169 (90) 2 AREE
# Aedes detritus: 135 (25)

# Provenienze (sources) da usare in test
test_sources = {"Aedes aegypti": ['Sud Africa'],#['ISS'],
                "Aedes albopictus": ['Berni', 'Guardavalle, Catanzaro', 'Montecchio, Terni', 'Sconosciuta', 'Sconosciuta_2'],
                "Aedes caspius": ['Voghera', 'Ferrara', 'centro ippico Grosseto'],
                "Aedes koreicus": ['ISS', 'Mezzolombardo, provincia di Trento (TN)', 'Sospirolo, provincia di Belluno (BL,VEN)'],
                "Anopheles maculipennis": ['Novara', 'località Berni,  provincia di Trento (TN)'],
                "Culex pipiens": ['IZS Sardegna', "San Michele all' Adige, provincia di Trento (TN)", \
                                  'Università degli Studi di Milano', 'Valcannuta'],
                "Culiseta longiareolata": ['Sconosciuta', 'Isola del Giglio', \
                                           'Spoleto', 'Ospedale veterinario didattico, Matelica, MC'],
                "Aedes vexans": ['Argenta', 'Sconosciuta_2', 'Bigarello', 'Concordia sulla secchia', 'Verona', \
                                 'Capriva del Friuli (GO)'],
                "Culex theileri ": [],
                "Anopheles algeriensis": ['Sconosciuta', 'Puglia '],
                "Aedes geniculatus": ['Fagagna (UD)', 'Gibellina,Trapani ?', 'Martellago', 'Piacenza', \
                                      "San Canzian d'Isonzo (GO)", 'Sconosciuta_9', 'Teramo'],
                "Aedes mariae": ['Custonaci', 'Marina di Lambrone', 'Sconosciuta'],
                "Culex hortensis": ['Arezzo', 'Dosoledo (BL)', 'Feltre', 'Pietrapiana, provincia di Firenze', \
                                    'Segni', 'Tivoli'],
                "Aedes rusticus": ['Parco Nazionale del Circeo', "Riserva Naturale dell'Insugherata"],
                "Aedes detritus": ['Lecce', 'Sconosciuta_2', 'Cabras', 'Sconosciuta']
}

# Aedes algeriensis e Aedes rusticus lasciate fuori dal training (usate solo in test)
# Culex theileri usate solo in training, in quanto ha una sola provenienza

# Lista di tutte le specie disponibili nel dataset al momento
tot_list = ["Aedes aegypti",
            "Aedes albopictus",
            "Aedes caspius",
            "Aedes koreicus",
            "Anopheles maculipennis",
            "Culex pipiens",
            "Aedes vexans",
            "Culiseta longiareolata",
            "Culex theileri",
            "Aedes geniculatus",
            "Aedes mariae",
            "Culex hortensis",
            "Anopheles algeriensis",
            "Aedes rusticus",
            #"Culiseta longiareolata",
            "Aedes detritus"]

# Dizionario che associa ad ogni specie un intero che fa da label
strig2num = {}
for i, s in enumerate(tot_list):
    strig2num[s] = i

# Operazione inversa. Dizionario che associa ad ogni intero una specie
num2strig = {value: key for key, value in strig2num.items()}
