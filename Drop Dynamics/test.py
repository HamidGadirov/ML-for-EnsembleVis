import json
with open('droplet-25-t10-feature-metadata.json', 'r') as f:
    labels = json.load(f)
    
#print("range:", labels["025_Hyspin-Hexadecan_0.26mm_viewA"]["features"][2])

print("len of features:", len(labels["025_Hyspin-Hexadecan_0.26mm_viewA"]["features"]))
len_of_f = len(labels["025_Hyspin-Hexadecan_0.26mm_viewA"]["features"])

for i in range (len_of_f): 
    print("features:", labels["025_Hyspin-Hexadecan_0.26mm_viewA"]["features"][i]["range"][0]) #range
    print("features:", labels["025_Hyspin-Hexadecan_0.26mm_viewA"]["features"][i]["name"]) #name
    
print("shape:", labels["025_Hyspin-Hexadecan_0.26mm_viewA"]["shape"][0])
