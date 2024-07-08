    ann_model_wheat = pickle.load(open("ann_model_wheat.pkl", "rb"))
    dist = {'24 Parganas': 1, 'Nadia': 2, 'Murshidabad': 3, 'Burdwan': 4, 'Birbhum': 5, 'Bankura': 6,
            'Hooghly': 7, 'Howrah': 8, 'Jalpaiguri': 9, 'Darjeeling': 10, 'Malda': 11, 'Cooch Behar': 12,
            'Purulia': 13, 'Midnapur': 14, 'West Dinajpur': 15}
    data["Name"] = request.form.get('name')
    data["Phone Number"] = request.form.get('phone')
    data["Email"] = request.form.get('email')
    data["State"] = request.form.get('state_opt')
    data["District"] = request.form.get('dist_opt')
    data["Type"] = request.form.get('type_loan')
    data["age"] = request.form.get('Year')
    num=dist[data["District"]]/15
    pre=ann_model_wheat.predict([num])
    pre=pre.reshape(-1)
    data["Predicted Production (1000 Tons)"]=pre[1]*1000
    data["Predicted Yield (kg/hactor)"]=pre[2]*10000
    data["Predicted Rainfall (cm)"]=pre[3]*1000
    data["Predicted irrigation (cm)"]=pre[4]*1000
    data["Predicted Min Av Temp (C)"]=pre[5]*100
    data["Predicted Max Av Temp (C)"]=pre[6]*100
    data["Predicted Soil Type"]=pre[7]
return render_template('Home.html', prediction_text='Entered data are {}'.format(data))