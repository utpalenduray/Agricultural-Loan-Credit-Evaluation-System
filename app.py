from flask import Flask, render_template, request, session, redirect, url_for
from decimal import Decimal, getcontext 
import pickle
import numpy as np
app = Flask(__name__)

app.secret_key = "utpalendu"

@app.route("/")
@app.route("/Home")
def hello_world():
    return render_template("Home.html")

@app.route('/personal')
def personal_page():
    return render_template("p_details.html")

@app.route('/agriculture')
def agriculture():
    return render_template("agriculture.html")

@app.route('/agriculture_dashboard')
def agriculture_dashboard():
    return render_template("agriculture_dashboard.html")


@app.route('/Personal', methods=['POST', 'GET'])
def Personal():
    if request.method == "POST":
        session["name"] = request.form["name"]
        session["phone"] = request.form["phone"]
        session["age"] = request.form["Year"]
        session["email"] = request.form["email"]
        session["state"] = request.form["state_opt"]
        session["district"] = request.form["dist_opt"]
        session["Existing_loan_check"]=request.form.get("Existing_loan_check", False)
        session["Existing_loan_check_EMI_Amount"]=request.form.get("Existing_loan_check_EMI_Amount", "0")
        session["purpose"] = request.form["purpose"]
        session["Annual_income"] = (request.form["Annual_income"])
        print(session["phone"])
    if "purpose" in session:
        purpose = session["purpose"]
        if purpose == "Agriculture":
            return redirect(url_for("agriculture"))
        elif purpose == "Life Stock Farming":
            return redirect(url_for("Life_Stock"))
        else:
            return redirect(url_for("Equipments"))
    else:
        return redirect(url_for("personal"))

@app.route('/agriculture_loan_details', methods=['POST', "GET"])
def agriculture_loan_details():
    if request.method == "POST":
        session["crop_type"] = request.form["crop_type1"]
        session["land_size"] =(request.form["land_size"])
        session["owner_land"] = request.form["owner_land"]
        session["irrigation_type"] = request.form["irrigation_type"]
        session["farming_type"] = request.form["farming_type"]
        session["Tractor"] = request.form.get("Tractor", False)
        session["Bulls"] = request.form.get("Bulls", False)
        session["Sprayer"] = request.form.get("Sprayer", False)
        session["Thresher"] = request.form.get("Thresher", False)
        session["family_labour"] = (request.form["family_labour"])
        session["repayment_type"] = request.form["repayment_type"]
    data=session
    #ANN Model for Production
    ann_model_wheat = pickle.load(open("ann_model_wheat.pkl", "rb"))
    dist = {'24 Parganas': 1, 'Nadia': 2, 'Murshidabad': 3, 'Burdwan': 4, 'Birbhum': 5, 'Bankura': 6,
            'Hooghly': 7, 'Howrah': 8, 'Jalpaiguri': 9, 'Darjeeling': 10, 'Malda': 11, 'Cooch Behar': 12,
            'Purulia': 13, 'Midnapur': 14, 'West Dinajpur': 15}
    num=dist[session["district"]]/15
    pre=ann_model_wheat.predict([num])
    pre=pre.reshape(-1)
    session["Predicted Production (1000 Tons)"]=round(pre[1]*1000)
    session["Predicted Yield (kg/hactor)"]=round(pre[2]*10000)
    session["Predicted Rainfall (cm)"]=round(pre[3]*1000)
    session["Predicted irrigation (cm)"]=round(pre[4]*1000)
    session["Predicted Min Av Temp (C)"]=round(15)
    session["Predicted Max Av Temp (C)"]=round(pre[6]*100)
    session["Predicted Soil Type"]=round(pre[7]*10)

    #LSMT Model for land Prepartion cost
    LSTM_model_wheat = pickle.load(open("lstm_land_prep_model_wheat.pkl", "rb"))
    last_year_cost=10083.633404/10000
    LSTM_wheat=LSTM_model_wheat.predict([[last_year_cost]])
    LSTM_wheat=LSTM_wheat.flatten()
    LSTM_wheat=float(LSTM_wheat)*10000
    land_cost=LSTM_wheat*float(request.form["land_size"])
    if (session["Tractor"]==True and session["Bulls"]==True):
        land_cost=land_cost-land_cost*0.15
    elif (session["Tractor"]==True and session["Bulls"]==False):
        land_cost=land_cost-land_cost*0.1
    elif (session["Tractor"]==False and session["Bulls"]==True):
        land_cost=land_cost-land_cost*0.1
    session["Land_preparation_cost"]=round(land_cost) 
    print(session["Land_preparation_cost"])

    #LSMT Model for seed cost
    LSTM_model_seed_price=pickle.load(open("lstm_seed_price_model_wheat.pkl", "rb"))
    last_year_cost_seed= 2963.036893/10000
    LSTM_seed=LSTM_model_seed_price.predict([[last_year_cost_seed]])
    LSTM_seed=LSTM_seed.flatten()
    LSTM_seed=float(LSTM_seed)*10000
    seed_cost=LSTM_seed*float(request.form["land_size"])
    session["Seed_price"]=round(seed_cost);
    print(session["Seed_price"])

    #LSMT Model for fertilizer cost
    LSTM_model_fert_price=pickle.load(open("lstm_fert_price_model_wheat.pkl", "rb"))
    last_year_fert_seed= 6127.746453/10000
    LSTM_fert=LSTM_model_fert_price.predict([[last_year_fert_seed]])
    LSTM_fert=LSTM_fert.flatten()
    LSTM_fert=float(LSTM_fert)*10000
    fert_cost=LSTM_fert*float(request.form["land_size"])
    session["Fertilizer_price"]=round(fert_cost);
    print(session["Fertilizer_price"])

    #LSMT Model for seed swoing cost
    LSTM_model_seed_swoing_price=pickle.load(open("lstm_fert_price_model_wheat.pkl", "rb"))
    last_year_seed_swing= 2792.390789/10000
    LSTM_seed_swoing=LSTM_model_fert_price.predict([[last_year_seed_swing]])
    LSTM_seed_swoing=LSTM_seed_swoing.flatten()
    LSTM_seed_swoing=float(LSTM_seed_swoing)*10000
    seed_swoing_cost=LSTM_seed_swoing*float(request.form["land_size"])
    if (eval(session["family_labour"])>=1 and eval(session["family_labour"])<=3):
        seed_swoing_cost=seed_swoing_cost-seed_swoing_cost*0.08*eval(session["family_labour"])
    elif (eval(session["family_labour"])>3):
        seed_swoing_cost=seed_swoing_cost-seed_swoing_cost*0.08*4
    session["Seed_swoing_cost"]=round(seed_swoing_cost);
    print(session["Seed_swoing_cost"])

    #LSMT Model for Irrigation cost
    LSTM_model_irrigation_price=pickle.load(open("lstm_irrigation_price_model_wheat.pkl", "rb"))
    last_year_irrigation= 6515.578507/10000
    LSTM_irrigation=LSTM_model_irrigation_price.predict([[last_year_irrigation]])
    LSTM_irrigation=LSTM_irrigation.flatten()
    LSTM_irrigation=float(LSTM_irrigation)*10000
    seed_irrigation_cost=LSTM_irrigation*float(request.form["land_size"])
    session["Irrigation_cost"]=round(seed_irrigation_cost);
    print(session["Irrigation_cost"])

    #LSMT Model for pesticide cost
    LSTM_model_pesticide_price=pickle.load(open("lstm_pesticide_price_model_wheat.pkl", "rb"))
    last_year_pesticide= 4653.984648/10000
    LSTM_pesticide=LSTM_model_pesticide_price.predict([[last_year_pesticide]])
    LSTM_pesticide=LSTM_pesticide.flatten()
    LSTM_pesticide=float(LSTM_pesticide)*10000
    seed_pesticide_cost=LSTM_pesticide*float(request.form["land_size"])
    if (session["Sprayer"]==True):
        seed_pesticide_cost=seed_pesticide_cost-seed_pesticide_cost*0.05
    session["Pesticide_cost"]=round(seed_pesticide_cost);
    print(session["Pesticide_cost"])

    #LSTM Model for Harvesting Cost
    LSTM_model_harvest_price=pickle.load(open("lstm_harvest_price_model_wheat.pkl", "rb"))
    last_year_harvest= 4343.719005/10000
    LSTM_harvest=LSTM_model_harvest_price.predict([[last_year_harvest]])
    LSTM_harvest=LSTM_harvest.flatten()
    LSTM_harvest=float(LSTM_harvest)*10000
    seed_harvest_cost=LSTM_harvest*float(request.form["land_size"])
    if (session["Thresher"]==True):
        seed_harvest_cost=seed_harvest_cost-seed_harvest_cost*0.15
    if (eval(session["family_labour"])>=1 and eval(session["family_labour"])<=3):
        seed_harvest_cost=seed_harvest_cost-seed_harvest_cost*0.08*eval(session["family_labour"])
    elif (eval(session["family_labour"])>3):
        seed_harvest_cost=seed_harvest_cost-seed_harvest_cost*0.08*4
    session["Harvest_cost"]=round(seed_harvest_cost);
    print(session["Harvest_cost"])

    #LSMT Model for Transport cost
    LSTM_model_transport_price=pickle.load(open("lstm_transport_price_model_wheat.pkl", "rb"))
    last_year_transport= 1861.593859/10000
    LSTM_transport=LSTM_model_transport_price.predict([[last_year_transport]])
    LSTM_transport=LSTM_transport.flatten()
    LSTM_transport=float(LSTM_transport)*10000
    seed_transport_cost=LSTM_transport*float(request.form["land_size"])
    if (session["Tractor"]==True and session["Bulls"]==True):
        seed_transport_cost=seed_transport_cost-seed_transport_cost*0.1
    session["Transport_cost"]=round(seed_transport_cost);
    print(session["Transport_cost"])
    
    # total cost of production
    total=session["Transport_cost"]+session["Land_preparation_cost"]+session["Seed_price"]+session["Harvest_cost"]+session["Pesticide_cost"]+session["Irrigation_cost"]+session["Seed_swoing_cost"]+session["Fertilizer_price"]
    session["Total_cost"]=total
    print(session["Total_cost"])

    labels_pie_prod=["Land_preparation_cost", "Seed_price", "Seed_swoing_cost","Irrigation_cost", "Pesticide_cost","Transport_cost", "Fertilizer_price", "Harvest_cost"]
    values_pie_prod=[session[a] for a in labels_pie_prod]
    labels_pie_prod=["Land Preparation", "Seed", "Seed Sowing","Irrigation", "Pesticide","Transport", "Fertilizer", "Harvest"]
    quantity=(session["Predicted Yield (kg/hactor)"])*float(request.form["land_size"])
    print(session["Predicted Yield (kg/hactor)"])
    print(session["land_size"])
    land_Size=float(request.form["land_size"])
    print( quantity)
    selling_price=round((Decimal(quantity)/Decimal(100))*2125)
    print(selling_price)
    getcontext().prec = 3
    roi=Decimal((selling_price-(session["Total_cost"]))/Decimal(session["Total_cost"]))*100
    print(roi)
    total_cost=session["Total_cost"]
    (net_profit)=round((selling_price)-(total_cost))
    print(session["Existing_loan_check_EMI_Amount"])
    print(session["Existing_loan_check"])
    ## ANN interval for wheat
    def calculate(pred_land_prep_cost, pred_seed_cost, pred_seed_sowing_cost, pred_irrigation_cost, pred_presticide_cost,pred_harvest_cost, pred_fertilizer_cost, pred_transport_cost):
        initial_amount = pred_land_prep_cost + pred_seed_cost + pred_seed_sowing_cost # Initial amount
        irrigation_per_interval = pred_irrigation_cost / 6
        fertiliser_per_interval = pred_fertilizer_cost / 6
        pesticides_per_interval = pred_presticide_cost / 3
        first_interval = irrigation_per_interval + fertiliser_per_interval # After 20 to 25 days of sowing
        second_interval = irrigation_per_interval + fertiliser_per_interval # After 40 to 45 days of sowing
        third_interval = irrigation_per_interval + fertiliser_per_interval # After 60 to 65 days of sowing
        fourth_interval = irrigation_per_interval + fertiliser_per_interval + pesticides_per_interval # After 80 to 85 days of sowing
        fifth_interval = irrigation_per_interval + fertiliser_per_interval + pesticides_per_interval # After 100 to 105 days of sowing
        sixth_interval = irrigation_per_interval + fertiliser_per_interval + pesticides_per_interval # After 105 to 120 days of sowing
        seventh_interval = pred_harvest_cost # After 120 days as required
        eighth_interval = pred_transport_cost # After harvesting as required
        return initial_amount,first_interval,second_interval,third_interval,fourth_interval,fifth_interval,sixth_interval,seventh_interval,eighth_interval

    initial_amount,first_interval,second_interval,third_interval,fourth_interval,fifth_interval,sixth_interval,seventh_interval,eighth_interval=calculate(session["Land_preparation_cost"],session["Seed_price"],session["Seed_swoing_cost"],session["Irrigation_cost"], session["Pesticide_cost"], session["Harvest_cost"],session["Fertilizer_price"],session["Transport_cost"])
    session["Min_amount"]=round(session["Total_cost"]*0.3)
    session["starting_amount"]=round(initial_amount-session["Total_cost"]*0.3)
    session["disbursement1"]=round(first_interval)
    session["disbursement2"]=round(second_interval)
    session["disbursement3"]=round(third_interval)
    session["disbursement4"]=round(fourth_interval)
    session["disbursement5"]=round(fifth_interval)
    session["disbursement6"]=round(sixth_interval)
    session["disbursement7"]=round(seventh_interval)
    session["disbursement8"]=round(eighth_interval)
    to=0
    labels_disburcement=["starting_amount", "disbursement1", "disbursement2","disbursement3","disbursement4","disbursement5","disbursement6","disbursement7","disbursement8"]
    labels_disburcement1=["starting_amount", "disbursement1", "disbursement2","disbursement3","disbursement4","disbursement5","disbursement6","disbursement7","disbursement8"]
    for x in labels_disburcement:
        to=to+session[x]
    amounts=[]
    amounts.append(to)
    for i in range(9):
        to=to-session[labels_disburcement[i]]
        amounts.append(to)
    labels_disburcement=["Allocated", "20th Day", "40th Day","60th Day","80th Day","100th Day","105th Day","120th Day","After Harvesting"]
    print(amounts)

    # emi amount and intrest calulation 
    def compound_interest(principal, rate, time):
        Amount = principal * (pow((1 + rate / 100), time))
        return Amount

    def calEMI_intrest(disbursement_amount, tenure_payback, interest_rate, gap):
        initail_amount=sum(disbursement_amount)
        disbursement_amount1=disbursement_amount.copy()
        disbursement_amount1[0]=compound_interest(disbursement_amount1[0],interest_rate,tenure_payback+gap)
        disbursement_amount1[1]=compound_interest(disbursement_amount1[1],interest_rate,tenure_payback-20/30+gap)
        disbursement_amount1[2]=compound_interest(disbursement_amount1[2],interest_rate,tenure_payback-40/30+gap)
        disbursement_amount1[3]=compound_interest(disbursement_amount1[3],interest_rate,tenure_payback-60/30+gap)
        disbursement_amount1[4]=compound_interest(disbursement_amount1[4],interest_rate,tenure_payback-80/30+gap)
        disbursement_amount1[5]=compound_interest(disbursement_amount1[5],interest_rate,tenure_payback-100/30+gap)
        disbursement_amount1[6]=compound_interest(disbursement_amount1[6],interest_rate,tenure_payback-105/30+gap)
        disbursement_amount1[7]=compound_interest(disbursement_amount1[7],interest_rate,tenure_payback-120/30+gap)
        total_payable_amount=sum(disbursement_amount1)
        emi=round(total_payable_amount/tenure_payback)
        intrest=round(total_payable_amount-initail_amount)
        return emi, intrest

    disbursement_amount=[]
    for i in range(9):
        disbursement_amount.append(session[labels_disburcement1[i]])
    
    tenure_payback=float(request.form["num_months"])
    interest_rate=12/12
    gap=0
    if session["repayment_type"]=="After Harvesting of Crop":
        gap=4
    (emi, intrest_paid)=calEMI_intrest(disbursement_amount, tenure_payback, interest_rate, gap)
    emilist=[]
    emilist.append(amounts[0])
    emilist.append(intrest_paid)
    print(emi)
    print(intrest_paid)
    print(disbursement_amount)
    return render_template('agriculture_dashboard.html',emilist=emilist, amounts=amounts,labels_disburcement=labels_disburcement, net_profit=net_profit,labels1=labels_pie_prod, values1=values_pie_prod, total_cost=total_cost, district=session["district"],selling_price=selling_price, roi=roi, quantity=quantity,land_Size=land_Size,  data=session, emi=emi, intrest_paid=intrest_paid, tenure_payback=tenure_payback)

if __name__ == "__main__":
    app.run(debug=True)
