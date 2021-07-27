#open with anaconda command prompt
#cd C:\Users\Ann\OneDrive\Desktop\fyp\Coding
#streamlit run deploy.py

import streamlit as st
import numpy as np
import pandas as pd
import datetime
import dateutil
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.ensemble import IsolationForest
import streamlit.components.v1 as components
import scipy.integrate  
import plotly.graph_objects as go
from plotly import tools
import plotly.offline as py
import plotly.express as px


st.set_option('deprecation.showPyplotGlobalUse', False)
@st.cache
def load_data(nrows):

    df = pd.read_csv("WholeCountry.csv")
    df.columns = [x.upper() for x in df.columns]
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index("STATE", inplace = True)
    return df

df = load_data(3120)

def load_data2(nrows):

    se = pd.read_csv("SelangorMukim.csv")
    se.columns = [x.upper() for x in se.columns]
    se['DATE'] = pd.to_datetime(se['DATE'])
    se.set_index("DISTRICT", inplace = True)
    return se

se = load_data2(1602)

st.title('HIGH RISK AREA PREDICTION FOR COVID-19 IN MALAYSIA (A SECOND OPINION)')
st.header('1.0 Exploratory Data Analysis')
st.text('Data in this section varies from March 2020 till Sep 2020 only')
select = st.radio('Select dataset:', ('Whole Country', 'Selangor Mukim'))
if select == 'Whole Country':
    st.write(df)
elif select == 'Selangor Mukim':
    st.write(se)

st.text("")
st.text("")
st.text("")

if st.checkbox('View 1st Dataset: Whole Country'):

    st.subheader('Cases in Each State from March till September')
    sns.set(rc={'figure.figsize':(30, 15)})
    ax = sns.swarmplot(x=df.index.values, y='NUMBEROFCONFIRMEDCASES', data=df)
    ax.set(ylim=(0, 80))
    st.pyplot(ax=ax) 
#############
    df3 = df.copy()
    #g = df3.groupby(pd.Grouper(freq="M"))  # DataFrameGroupBy (grouped by Month)
    #df3.set_index('DATE').index.astype('datetime64[ns]')
    df3['MONTH'] = df3['DATE'].dt.month 
    abc = df3[(df3['MONTH']<10) & (df3['MONTH']>2)]
    abc['MONTH'].unique().tolist()

    st.subheader('Cases in Each State by Month')
    fig, ax = plt.subplots(figsize=(25, 20))
    ax = sns.swarmplot(data=abc, x="MONTH", y="NUMBEROFCONFIRMEDCASES",  ax=ax)
    ax.set(ylim=(0, 100))
    st.pyplot(ax=ax)
############
    st.subheader('Total Cases in Each State from March till September')
    fig, ax = plt.subplots(1, figsize=(16,10))
    df.groupby(["STATE"]).sum().plot(kind='bar')
    st.pyplot()
#############

    df2 = df.copy()
    df2 = df2.groupby(pd.Grouper(key='DATE',freq='M')).mean()
    df2 = df2.iloc[2:9]

    st.subheader("Mean of Confirmed Cases by Month")
    fig, ax = plt.subplots()
    df2.plot(figsize=(10, 6))
    st.pyplot()



if st.checkbox('View 2nd Dataset: Selangor Mukim'):
    
    se3 = se.copy()
    se3['MONTH'] = se3['DATE'].dt.month 
    abc2 = se3[(se3['MONTH']<10) & (se3['MONTH']>2)]
    st.subheader("Number of Cases in Township of Selangor by Month")
    fig, ax = plt.subplots(figsize=(25, 20))
    ax = sns.swarmplot(data=abc2, x="MONTH", y="NUMBEROFCUMULATIVECASES",  ax=ax)
    st.pyplot(ax=ax)
############    
    st.subheader("Number of Cases in Township of Selangor from March till September")
    sns.set(rc={'figure.figsize':(50, 15)})
    ax = sns.swarmplot(x=se.index.values, y='NUMBEROFCUMULATIVECASES', data=se)
    ax.set(ylim=(0, 800))
    st.pyplot()
############
    fig1 = plt.figure(figsize=(10,6))
    plt.hist(se["NUMBEROFCUMULATIVECASES"], bins=[0,10,30,50,70,90,130,150,180,210,250,280,310,350,380,
                                              410,450,510,610,630,650])
    plt.style.use('ggplot')
    plt.title("Range of Cumulative Cases in Selangor from March till Sep")
    plt.xlabel('Number of Cumulative Cases')
    st.pyplot(fig1)
#############
    se2 = se.copy()
    se2 = se2.groupby(pd.Grouper(key='DATE',freq='M')).mean()
    se2 = se2.iloc[2:9]

    st.subheader("Cumulative Average Cases by Month")
    fig, ax = plt.subplots()
    se2.plot(figsize=(10, 6))
    st.pyplot()

st.header('2.0 Rate of Change with Analysis')
roc = pd.read_csv("State.csv")
roc.columns = [x.upper() for x in roc.columns]
roc['DATE'] = pd.to_datetime(roc['DATE'])

roc['PERLIS'] = (roc['PERLIS'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))
roc['KEDAH'] = (roc['KEDAH'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))
roc['PULAUPINANG'] = (roc['PULAUPINANG'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))
roc['PERAK'] = (roc['PERAK'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))
roc['SELANGOR'] = (roc['SELANGOR'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))
roc['NEGERISEMBILAN'] = (roc['NEGERISEMBILAN'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))
roc['JOHOR'] = (roc['JOHOR'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))
roc['PAHANG'] = (roc['PAHANG'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))
roc['MELAKA'] = (roc['MELAKA'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))
roc['TERENGGANU'] = (roc['TERENGGANU'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))
roc['KELANTAN'] = (roc['KELANTAN'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))
roc['SABAH'] = (roc['SABAH'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))
roc['SARAWAK'] = (roc['SARAWAK'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))
roc['KUALALUMPUR'] = (roc['KUALALUMPUR'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))
roc['PUTRAJAYA'] = (roc['PUTRAJAYA'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))
roc['LABUAN'] = (roc['LABUAN'].transform(lambda s: s.sub(s.shift().fillna(0)).abs()))

roc['ROC_PERLIS'] = roc['PERLIS'].diff()
roc['ROC_KEDAH'] = roc['KEDAH'].diff()
roc['ROC_PULAUPINANG'] = roc['PULAUPINANG'].diff()
roc['ROC_PERAK'] = roc['PERAK'].diff()
roc['ROC_SELANGOR'] = roc['SELANGOR'].diff()
roc['ROC_NEGERISEMBILAN'] = roc['NEGERISEMBILAN'].diff()
roc['ROC_MELAKA'] = roc['MELAKA'].diff()
roc['ROC_JOHOR'] = roc['JOHOR'].diff()
roc['ROC_PAHANG'] = roc['PAHANG'].diff()
roc['ROC_TERENGGANU'] = roc['TERENGGANU'].diff()
roc['ROC_KELANTAN'] = roc['KELANTAN'].diff()
roc['ROC_SABAH'] = roc['SABAH'].diff()
roc['ROC_SARAWAK'] = roc['SARAWAK'].diff()
roc['ROC_KUALALUMPUR'] = roc['KUALALUMPUR'].diff()
roc['ROC_PUTRAJAYA'] = roc['PUTRAJAYA'].diff()
roc['ROC_LABUAN'] = roc['LABUAN'].diff()

roc2 = roc.copy()
roc2.set_index("DATE", inplace = True)
roc2 = roc2.drop(['PERLIS'], axis = 1)
roc2 = roc2.drop(['KEDAH'], axis = 1)
roc2 = roc2.drop(['PULAUPINANG'], axis = 1)
roc2 = roc2.drop(['PERAK'], axis = 1)
roc2 = roc2.drop(['SELANGOR'], axis = 1)
roc2 = roc2.drop(['NEGERISEMBILAN'], axis = 1)
roc2 = roc2.drop(['MELAKA'], axis = 1)
roc2 = roc2.drop(['JOHOR'], axis = 1)
roc2 = roc2.drop(['PAHANG'], axis = 1)
roc2 = roc2.drop(['TERENGGANU'], axis = 1)
roc2 = roc2.drop(['KELANTAN'], axis = 1)
roc2 = roc2.drop(['SABAH'], axis = 1)
roc2 = roc2.drop(['SARAWAK'], axis = 1)
roc2 = roc2.drop(['KUALALUMPUR'], axis = 1)
roc2 = roc2.drop(['PUTRAJAYA'], axis = 1)
roc2 = roc2.drop(['LABUAN'], axis = 1)

plt.figure(figsize=[15,10])
plt.grid(True)
plt.plot(roc2['ROC_PERLIS'], label = 'PERLIS', color = 'red')
plt.plot(roc2['ROC_KEDAH'], label = 'KEDAH', color = 'orange')
plt.plot(roc2['ROC_PULAUPINANG'], label = 'PULAUPINANG', color = 'yellow')
plt.plot(roc2['ROC_PERAK'], label = 'PERAK', color = '#1890df')
plt.plot(roc2['ROC_SELANGOR'], label = 'SELANGOR', color = 'green')
plt.plot(roc2['ROC_NEGERISEMBILAN'], label = 'NEGERISEMBILAN', color = 'cyan')
plt.plot(roc2['ROC_MELAKA'], label = 'MELAKA', color = 'blue')
plt.plot(roc2['ROC_JOHOR'], label = 'JOHOR', color = 'purple')
plt.plot(roc2['ROC_PAHANG'], label = 'PAHANG', color = 'deeppink')
plt.plot(roc2['ROC_TERENGGANU'], label = 'TERENGGANU', color = '#873ae1')
plt.plot(roc2['ROC_KELANTAN'], label = 'KELANTAN', color = 'darkgray')
plt.plot(roc2['ROC_SABAH'], label = 'SABAH', color = '#2fe289')
plt.plot(roc2['ROC_SARAWAK'], label = 'SARAWAK', color = 'wheat')
plt.plot(roc2['ROC_KUALALUMPUR'], label = 'KUALALUMPUR', color = '#9450e0')
plt.plot(roc2['ROC_PUTRAJAYA'], label = 'PUTRAJAYA', color = 'black')
plt.plot(roc2['ROC_LABUAN'], label = 'LABUAN', color = '#C67171')
plt.legend(loc=2)
plt.xticks([])
plt.title('Rate of Change of Cases by State', fontsize=20)
plt.xlabel('Date from 16/3/2020 - 11/1/2021', fontsize=16)
plt.ylabel('Number of Cases', fontsize=16)
st.pyplot()

st.subheader('Rate of Change of State and Percentage Change')
def load_data10(nrows):
    haha = pd.read_csv("pct_State.csv")
    return haha

haha = load_data10(303)   
st.write(haha)



st.header('3.0 SEIR Model')
st.text('***Due to lack of data sources of population that could be found online, each population ')
st.text('recorded in the year of 2010 is added with 2% to compromise with the latest data.')

def load_data3(nrows):
    popu = pd.read_csv("Population.csv")
    popu.set_index("District", inplace = True)
    return popu

popu = load_data3(11)   
st.write(popu)

user_input1 = st.number_input("Population of District")
if user_input1 > 0:
    def seir(x,y):
            def base_seir_model(init_vals, params, t):
                S_0, E_0, I_0, R_0 = init_vals
                S, E, I, R = [S_0], [E_0], [I_0], [R_0]
                alpha, beta, gamma = params
                dt = t[1] - t[0]
                for _ in t[1:]:
                    next_S = S[-1] - (beta*S[-1]*I[-1])*dt
                    next_E = E[-1] + (beta*S[-1]*I[-1] - alpha*E[-1])*dt
                    next_I = I[-1] + (alpha*E[-1] - gamma*I[-1])*dt
                    next_R = R[-1] + (gamma*I[-1])*dt
                    S.append(next_S)
                    E.append(next_E)
                    I.append(next_I)
                    R.append(next_R)
                return np.stack([S, E, I, R]).T

            # Define parameters
            t_max = 87
            dt = 1
            t = np.linspace(0, t_max, int(t_max/dt) + 1)
            N = user_input1
            init_vals = 1 - 1/N, 1/N, 0, 0
            alpha = 0.2
            beta = x*0.4 #R0*gamma
            gamma = 0.4 
            params = alpha, beta, gamma
            # Run simulation
            results = base_seir_model(init_vals, params, t)


            df_SEIR_1 = pd.DataFrame(results, columns=['Susceptible', 'Exposed', 'Infected', 'Removed'])
            df_SEIR_1['Day'] = 1 + df_SEIR_1.index
            df_SEIR_1['Date']=pd.date_range(start=y, periods=t_max+1)
            df_SEIR_1_melt = pd.melt(df_SEIR_1, id_vars=['Date'], value_vars=['Susceptible', 'Exposed', 'Infected', 'Removed'])  
            df_SEIR_1_melt.value = round(df_SEIR_1_melt.value * N,0)

            
            alt.data_transformers.disable_max_rows()
            source = df_SEIR_1_melt
            #source = df_melt[df_melt.variable.isin(['E','I'])]


            nearest = alt.selection(type='single', nearest=True, on='mouseover',
                                    fields=['Date'], empty='none')


            # The basic line
            line = alt.Chart(source, title="Social Distancing and Without Vital Dynamic").mark_line().encode(
                x= alt.X('Date', title='Day'),
                y= alt.Y('value', title='% of Population'),   
                color='variable' 
            )
            # Transparent selectors across the chart. This is what tells us
            # the x-value of the cursor
            selectors = alt.Chart(source).mark_point().encode(
                x='Date',opacity=alt.value(0),).add_selection(nearest).interactive()

            # Draw points on the line, and highlight based on selection
            points = line.mark_point().encode(
                opacity=alt.condition(nearest, alt.value(1), alt.value(0))
            )

            # Draw text labels near the points, and highlight based on selection
            text = line.mark_text(align='left', dx=5, dy=-5).encode(
                text=alt.condition(nearest, 'value', alt.value(' '))
            )

            # Draw a rule at the location of the selection
            rules = alt.Chart(source).mark_rule(color='gray').encode(
                x='Date',
            ).transform_filter(
                nearest
            )

            # Put the five layers into a chart and bind the data
            seir1 = alt.layer(line, selectors, points, rules, text)
            chart = st.altair_chart(alt.layer(seir1).properties(width=700,height=550).interactive())
            return chart
    
    select = st.radio('Select wave of Covid-19:', ('2nd wave', '3rd wave'))
    if select == '2nd wave':
        x = 3.5
        y = (datetime.datetime(2020, 4, 6))
        seir(x,y)
    if select == '3rd wave':
        x = 2.3
        y = (datetime.datetime(2020, 9, 8))
        seir(x,y)
     
if user_input1 < 0 :
    st.write("Please enter a valid population number")


st.header('4.0 Simple Moving Average')


#def load_data6(nrows):
#    ccc = pd.read_csv("sma.csv")
#    return ccc
    
#ccc = load_data6(76) 

#st.write(ccc)


def load_data4(nrows):
    distr = pd.read_csv("District.csv")
    return distr
    
distr = load_data4(9) 
st.text('Types of District List: ')
st.write(distr)

array = ("Gombak", "HuluLangat","HuluSelangor","Klang",
"KualaLangat","KualaSelangor","Petaling","SabakBernam","Sepang")


se = load_data2(1602)
mask = se.copy()

if st.checkbox('2nd wave'):
    mask1 = (mask['DATE'] > '5/4/2020') & (mask['DATE'] <= '30/6/2020')
    mask1 = mask.loc[mask1]
    def sma(nrows):
        sma = mask1.copy()
        sma['pandas_SMA_3'] = sma.iloc[:,-1].rolling(window=3).mean()
        sma['pandas_SMA_4'] = sma.iloc[:,-1].rolling(window=4).mean()
        sma['pandas_SMA_5'] = sma.iloc[:,-1].rolling(window=5).mean()
        return sma
    
    sma_ = sma(9)
    smaplot = sma_.groupby(pd.Grouper(key='DATE',freq='w')).mean()
    smaplot = smaplot.dropna()

    fig = plt.figure(figsize=[15,10])
    plt.grid(True)
    plt.plot(smaplot['NUMBEROFCUMULATIVECASES'], label=' Real Data ')
    plt.plot(smaplot['pandas_SMA_3'], label=' SMA 3 ')
    plt.plot(smaplot['pandas_SMA_4'], label=' SMA 4 ')
    plt.plot(smaplot['pandas_SMA_5'], label=' SMA 5 ')
    plt.legend(loc=2)
    plt.title('Simple Moving Average', fontsize=20)
    plt.xlabel('DATE', fontsize=16)
    plt.ylabel('Number of Cases', fontsize=16)

    def evaluation():   
        yhat = final['pandas_SMA_3']
        zhat = final['pandas_SMA_4']
        hat = final['pandas_SMA_5']
        ori = final['NUMBEROFCUMULATIVECASES_x']
        d = ori - yhat
        e = ori - zhat
        f = ori - hat
        mse_f1 = np.mean(d**2)
        mae_f1 = np.mean(abs(d))
        rmse_f1 = np.sqrt(mse_f1)
        mse_f2 = np.mean(e**2)
        mae_f2 = np.mean(abs(e))
        rmse_f2 = np.sqrt(mse_f2)
        mse_f3 = np.mean(f**2)
        mae_f3 = np.mean(abs(f))
        rmse_f3 = np.sqrt(mse_f3)

        st.write("MAE_3:",mae_f1)
        st.write("MSE_3:", mse_f1)
        st.write("RMSE_3:", rmse_f1)
        st.write("MAE_4:",mae_f2)
        st.write("MSE_4:", mse_f2)
        st.write("RMSE_4:", rmse_f2)
        st.write("MAE_5:",mae_f3)
        st.write("MSE_5:", mse_f3)
        st.write("RMSE_5:", rmse_f3)

    user_input3 = st.text_input("Please enter a district name")
    if user_input3 == array[0] :
        mask1 = mask1.loc['Gombak']
        final = pd.merge(smaplot, mask1, on=['DATE'])
        final.drop('NUMBEROFCUMULATIVECASES_y', axis=1, inplace=True)
        st.pyplot(fig)
        evaluation()
    pass
    if user_input3 == array[1] :
        mask1 = mask1.loc['HuluLangat']
        final = pd.merge(smaplot, mask1, on=['DATE'])
        final.drop('NUMBEROFCUMULATIVECASES_y', axis=1, inplace=True)
        st.pyplot(fig)
        evaluation()
    pass
    if user_input3 == array[2] :
        mask1 = mask1.loc['HuluSelangor']
        final = pd.merge(smaplot, mask1, on=['DATE'])
        final.drop('NUMBEROFCUMULATIVECASES_y', axis=1, inplace=True)
        st.pyplot(fig)
        evaluation()
    pass
    if user_input3 == array[3] :
        mask1 = mask1.loc['Klang']
        final = pd.merge(smaplot, mask1, on=['DATE'])
        final.drop('NUMBEROFCUMULATIVECASES_y', axis=1, inplace=True)
        st.pyplot(fig)
        evaluation()
    pass
    if user_input3 == array[4] :
        mask1 = mask1.loc['KualaLangat']
        final = pd.merge(smaplot, mask1, on=['DATE'])
        final.drop('NUMBEROFCUMULATIVECASES_y', axis=1, inplace=True)
        st.pyplot(fig)
        evaluation()
    pass
    if user_input3 == array[5] :
        mask1 = mask1.loc['KualaSelangor']
        final = pd.merge(smaplot, mask1, on=['DATE'])
        final.drop('NUMBEROFCUMULATIVECASES_y', axis=1, inplace=True)
        st.pyplot(fig)
        evaluation()
    pass
    if user_input3 == array[6] :
        mask1 = mask1.loc['Petaling']
        final = pd.merge(smaplot, mask1, on=['DATE'])
        final.drop('NUMBEROFCUMULATIVECASES_y', axis=1, inplace=True)
        st.pyplot(fig)
        evaluation()
    pass
    if user_input3 == array[7] :
        mask1 = mask1.loc['SabakBernam']
        final = pd.merge(smaplot, mask1, on=['DATE'])
        final.drop('NUMBEROFCUMULATIVECASES_y', axis=1, inplace=True)
        st.pyplot(fig)
        evaluation()
    pass
    if user_input3 == array[8] :
        mask1 = mask1.loc['Sepang']
        final = pd.merge(smaplot, mask1, on=['DATE'])
        final.drop('NUMBEROFCUMULATIVECASES_y', axis=1, inplace=True)
        st.pyplot(fig)
        evaluation()
    pass

if st.checkbox('3rd wave'):
    wave3 = (mask['DATE'] > '8/9/2020') & (mask['DATE'] <= '4/1/2021')
    wave3 = mask.loc[wave3]
    fig2 = plt.figure(figsize=[15,10])
    plt.grid(True)
    plt.plot(smaplot['NUMBEROFCUMULATIVECASES'], label=' Real Data ')
    plt.plot(smaplot['pandas_SMA_3'], label=' SMA 3 ')
    plt.plot(smaplot['pandas_SMA_4'], label=' SMA 4 ')
    plt.plot(smaplot['pandas_SMA_5'], label=' SMA 5 ')
    plt.legend(loc=2)
    plt.title('Simple Moving Average', fontsize=20)
    plt.xlabel('DATE', fontsize=16)
    plt.ylabel('Number of Cases', fontsize=16)
    st.pyplot(fig2)

    def plot2(nrows):
        sma3rd = wave3.copy()
        sma3rd['pandas_SMA_3'] = sma3rd.iloc[:,-1].rolling(window=3).mean()
        sma3rd['pandas_SMA_4'] = sma3rd.iloc[:,-1].rolling(window=4).mean()
        sma3rd['pandas_SMA_5'] = sma3rd.iloc[:,-1].rolling(window=5).mean()
        smaplot3 = sma3rd.groupby(pd.Grouper(key='DATE',freq='w')).mean()
        smaplot3 = smaplot3.dropna()
        return smaplot3
    smaplot3 = plot2(9)
    
    fig3rd = plt.figure(figsize=[15,10])
    plt.grid(True)
    plt.plot(smaplot3['NUMBEROFCUMULATIVECASES'], label=' Real Data ')
    plt.plot(smaplot3['pandas_SMA_3'], label=' SMA 3 ')
    plt.plot(smaplot3['pandas_SMA_4'], label=' SMA 4 ')
    plt.plot(smaplot3['pandas_SMA_5'], label=' SMA 5 ')
    plt.legend(loc=2)
    plt.title('Simple Moving Average', fontsize=20)
    plt.xlabel('DATE', fontsize=16)
    plt.ylabel('Number of Cases', fontsize=16)

    def evaluation2():   
        yhat = smaplot3['pandas_SMA_3']
        zhat = smaplot3['pandas_SMA_4']
        hat = smaplot3['pandas_SMA_5']
        ori = smaplot3['NUMBEROFCUMULATIVECASES']
        oy = ori - yhat    
        of = ori - zhat
        od = ori - hat
        mse_f_1 = np.mean(oy**2)
        mae_f_1 = np.mean(abs(oy))
        rmse_f_1 = np.sqrt(mse_f_1)
        mse_f_2 = np.mean(of**2)
        mae_f_2 = np.mean(abs(of))
        rmse_f_2 = np.sqrt(mse_f_2)
        mse_f_3 = np.mean(od**2)
        mae_f_3 = np.mean(abs(od))
        rmse_f_3 = np.sqrt(mse_f_3)
        st.write("MAE_3:",mae_f_1)
        st.write("MSE_3:", mse_f_1)
        st.write("RMSE_3:", rmse_f_1)
        st.write("MAE_4:",mae_f_2)
        st.write("MSE_4:", mse_f_2)
        st.write("RMSE_4:", rmse_f_2)
        st.write("MAE_5:",mae_f_3)
        st.write("MSE_5:", mse_f_3)
        st.write("RMSE_5:", rmse_f_3)


    #x = 1
    #b1 = st.selectbox("1. Gombak", key="1")
    #b2 = st.selectbox("2. Hulu Langat", key="2")
    #b3 = st.selectbox("3. Hulu Selangor", key="3")
    #b4 = st.selectbox("4. Klang", key="4")
    #b5 = st.selectbox("5. Kuala Langat", key="5")
    #b6 = st.selectbox("6. Kuala Selangor", key="6")
    #b7 = st.selectbox("7. Petaling", key="7")
    #b8 = st.selectbox("8. Sabak Bernam", key="8")
    #b9 = st.selectbox("4. Labuan", key="9")
    #b = st.selectbox("Exit", key="50")

    option = st.selectbox('Select district',('Gombak','HuluLangat','HuluSelangor','Klang',
                            'KualaLangat','KualaSelangor','Petaling','SabakBernam','Labuan'))
    
    #while x > 0:
    if option == 'Gombak':
        smaplot3 = smaplot3.loc['Gombak']
        st.pyplot(fig3rd)
        evaluation2()
    pass

    if option == 'HuluLangat':
        smaplot3 = smaplot3.loc['HuluLangat']
        st.pyplot(fig3rd)
        evaluation2()
    pass

    if option == 'HuluSelangor':
        smaplot3 = smaplot3.loc['HuluSelangor']
        st.pyplot(fig3rd)
        evaluation2()
    pass

    if option == 'Klang':
        smaplot3 = smaplot3.loc['Klang']
        st.pyplot(fig3rd)
        evaluation2()
    pass

    if option == 'KualaLangat':
        smaplot3 = smaplot3.loc['KualaLangat']
        st.pyplot(fig3rd)
        evaluation2()
    pass

    if option == 'KualaSelangor':
        smaplot3 = smaplot3.loc['KualaSelangor']
        st.pyplot(fig3rd)
        evaluation2()
    pass

    if option == 'Petaling':
        smaplot3 = smaplot3.loc['Petaling']
        st.pyplot(fig3rd)
        evaluation2()
    pass

    if option == 'SabakBernam':
        smaplot3 = smaplot3.loc['SabakBernam']
        st.pyplot(fig3rd)
        evaluation2()
    pass

    if option == 'Labuan':
        smaplot3 = smaplot3.loc['Labuan']
        st.pyplot(fig3rd)
        evaluation2()
    pass



