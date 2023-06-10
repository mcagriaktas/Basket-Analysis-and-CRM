import pandas as pd


df = pd.read_csv('online_retail_II.csv')

"""----------------- Test Codes ------------------------------------------"""


"""---------------- Information of The dataset --------------------------"""

pd.set_option('display.max_columns', 8)

# Dataset's general information.
def columns_type():
    print(df.info())


# Sum of row and column. (The dataset was not cleaned.)
def row_column_sum():
    print("All the dataset's row and column of count: ", df.shape)
    df.dropna(inplace=True)
    print("Clean dataset's row and column of count: ", df.shape)


# Columns' names.
def columns_names():
    column_count = len(df.columns)
    for i in range(0, column_count):
        print(f"{i + 1}. column's name: ", df.columns[i])


# All null values.
def null_values():
    print(df.isnull().sum())


# Number of transactions by country.
def count_country():
    country = df['Country'].value_counts()
    print(country)


# Total of every parameter in the database.
def total_revenue():
    global df
    df.dropna(inplace=True)
    df = df[~df["Invoice"].str.contains("C", na=False)]

    products = df['Invoice'].nunique()
    invoice = df['Description'].nunique()
    revenue = round((df["Quantity"] * df["Price"]).sum(), 2)
    customer = df['Customer ID'].nunique()

    print("Products: ", products)
    print("Invoice: ", invoice)
    print("Revenue: ", revenue)
    print("Customer: ", customer)


"""------------------- The Data Cleaning for Analysis ---------------------"""

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C") &
        ~df["StockCode"].str.contains("POST") &
        ~df["StockCode"].str.contains("M") &
        ~df["StockCode"].str.contains("DOT", na=False) &
         (df["Quantity"] > 0) &
         (df["Price"] > 0)]

def outliner_cleaning():
    import matplotlib.pyplot as plt

    def outlier_thresholds(dataframe, variable):
        quartile1 = dataframe[variable].quantile(0.01)
        quartile3 = dataframe[variable].quantile(0.99)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return up_limit, low_limit

    def replace_with_threshold(dataframe, variable):
        up_limit, low_limit = outlier_thresholds(dataframe, variable)
        # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    # f, ax = plt.subplots(2,1, figsize = (20,5))
    # col_list = ["Quantity", "Price"]
    # # for i in range(0,2):
    # #     ax[i].boxplot(df[col_list[i]], vert = 0)
    # # plt.show()

    replace_with_threshold(df, "Quantity")
    replace_with_threshold(df, "Price")

    # f, ax = plt.subplots(2,1, figsize = (20,5))
    # col_list = ["Quantity", "Price"]
    # for i in range(0,2):
    #     ax[i].boxplot(df[col_list[i]], vert = 0)
    # plt.show()

outliner_cleaning()

print(df['Invoice'].nunique())
print(df['Description'].nunique())
print(df['Customer ID'].nunique())

country = 'France'
df_country = df[df['Country'] == country]
freq = df_country.groupby(['Invoice', 'Description'])['Quantity'].sum()
prod_freq = freq.unstack().fillna(0).reset_index().set_index('Invoice')




"""--------------------- Basket Analysis's Apriori Codes -----------------"""

def apriori():
    from mlxtend.frequent_patterns import fpgrowth
    from mlxtend.frequent_patterns import association_rules

    run_1 = prod_freq.applymap(lambda x: 1 if x > 0 else 0).astype(bool)

    total = run_1.shape[0]
    run_2 = ("Total numer of transactions for France: ", total)

    frequent_itemsets = fpgrowth(run_1, min_support=0.08, use_colnames=True)
    association_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    filtered_rules = association_rules[~association_rules['consequents'].apply(lambda x: 'POSTAGE' in x)]
    run_3 = filtered_rules[['antecedents', 'consequents', 'confidence']]
    run_3 = run_3.sort_values(by='confidence', ascending=False)
    run_3.to_excel("aprior_final.xlsx", index = False)


"""---------------------- RFM - (Recency, Frequency, Monetary)  -----------"""

def RFM():
    import datetime as dt
    import matplotlib.pyplot as plt
    import seaborn as sns

    df["Total_Prize"] = df["Quantity"] * df["Price"]
    df["InvoiceDate"] = df["InvoiceDate"].apply(pd.to_datetime)

    df["InvoiceDate"].max()
    today_date = dt.datetime(2011,12,11)

    df_rfm = df.groupby("Customer ID").agg({"InvoiceDate" : lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                            "Invoice" : lambda Invoice : Invoice.nunique(),
                                            "Total_Prize" : lambda Total_Prize: Total_Prize.sum()})

    df_rfm.columns = ["Recency", "Frequency", "Monetary"]

    df_rfm["Recency_Score"] = pd.qcut(df_rfm["Recency"], 5, labels = [5, 4, 3, 2, 1])
    df_rfm["Frequency_Score"] = pd.qcut(df_rfm["Frequency"].rank(method = "first"), 5, labels = [1, 2, 3, 4, 5])
    df_rfm["Monetary_Score"] = pd.qcut(df_rfm["Monetary"], 5, labels = [1, 2, 3, 4, 5])

    df_rfm["RF_Score"] = df_rfm["Recency_Score"].astype(str) + df_rfm["Frequency_Score"].astype(str)

    seg_map = {
        r'[1-2][1-2]': 'hibernatig',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant-loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'}

    df_rfm["segment"] = df_rfm["RF_Score"].replace(seg_map, regex = True)

    # sns.pairplot(df_rfm, hue = "segment")
    # plt.show()


"""---------------------- Simple Solution Program -------------------------"""


"""------ Data Base (Example)------------------------------------------- """
def program():
    import datetime as dt
    import matplotlib.pyplot as plt
    import seaborn as sns

    df["Total_Prize"] = df["Quantity"] * df["Price"]
    df["InvoiceDate"] = df["InvoiceDate"].apply(pd.to_datetime)

    df["InvoiceDate"].max()
    today_date = dt.datetime(2011, 12, 11)

    df_rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                            "Invoice": lambda Invoice: Invoice.nunique(),
                                            "Total_Prize": lambda Total_Prize: Total_Prize.sum()})

    df_rfm.columns = ["Recency", "Frequency", "Monetary"]

    df_rfm["Recency_Score"] = pd.qcut(df_rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
    df_rfm["Frequency_Score"] = pd.qcut(df_rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    df_rfm["Monetary_Score"] = pd.qcut(df_rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

    df_rfm["RF_Score"] = df_rfm["Recency_Score"].astype(str) + df_rfm["Frequency_Score"].astype(str)

    seg_map = {
        r'[1-2][1-2]': 'hibernatig',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant-loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'}

    df_rfm["segment"] = df_rfm["RF_Score"].replace(seg_map, regex=True)

    """----------------- PROGRAM MAIN CODES--------------------------"""

    while True:
        print("Please enter Customer ID"
              "If you want to exit, please enter '-1'")

        userid = int(input(":   "))

        if userid == 'q':
            break

        if userid in df_rfm.index:
            recency = df_rfm.loc[userid, "Recency"]
            frequency = df_rfm.loc[userid, "Frequency"]
            recency_score = df_rfm.loc[userid, "Recency_Score"]
            frequency_score = df_rfm.loc[userid, "Frequency_Score"]
            monetary_score = df_rfm.loc[userid, "Monetary_Score"]
            rf_score = df_rfm.loc[userid, "RF_Score"]
            segment = df_rfm.loc[userid, "segment"]

            print("Recency:", recency)
            print("Frequency:", frequency)
            print("Recency Score:", recency_score)
            print("Frequency Score:", frequency_score)
            print("Monetary Score:", monetary_score)
            print("RF Score:", rf_score)
            print("Segment:", segment)

            print("Müşteriye yapılacak öneriler:")

            if segment == "hibernating":
                print("- Uyandırmak için özel teklifler sunun.")
                print("- İlgilendikleri ürün ve hizmetler hakkında bilgilendirin.")

            elif segment == "at_Risk":
                print("- Müşteri memnuniyetini artırmak için geri bildirim alın.")
                print("- Sadakat programları ve indirimler sunun.")

            elif segment == "cant-lose":
                print("- Özel teşekkür notları ve hediyeler gönderin.")
                print("- Yeni ürünler ve hizmetler hakkında bilgilendirin.")

            elif segment == "about_to_sleep":
                print("- Yeniden ilgilerini çekmek için özel teklifler sunun.")
                print("- Kişiselleştirilmiş e-postalar ve kampanyalar gönderin.")

            elif segment == "need_attention":
                print("- Müşteriyle iletişim kurmak için özel bir temsilci atayın.")
                print("- Sorunları çözmek için hızlı müdahale sağlayın.")

            elif segment == "loyal_customers":
                print("- Sadakat programları ve ödüller sunun.")
                print("- VIP avantajları ve özel indirimler sağlayın.")

            elif segment == "promising":
                print("- Yeni ürünler ve hizmetler hakkında bilgilendirin.")
                print("- İlgilendikleri konularla ilgili içerikler sunun.")

            elif segment == "new_customers":
                print("- Hoş geldin indirimleri ve promosyonlar sunun.")
                print("- Ürün ve hizmetlerinizi tanıtmak için kişiselleştirilmiş kampanyalar oluşturun.")

            elif segment == "potential_loyalists":
                print("- Sadakat programlarına kaydolmaları için teşvik edin.")
                print("- Özel indirimler ve avantajlar sunun.")

            elif segment == "champions":
                print("- Özel teşekkür notları ve hediyeler gönderin.")
                print("- Özel indirimler ve promosyonlar sunun.")
                print("- Yeni ürünler ve hizmetler hakkında bilgilendirin.")

        else:
            print("Customer ID not found.")


"""------------------- Product Categorization (Not Finished) --------------"""

def most_40_words():
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from collections import Counter
    import matplotlib.pyplot as plt


    product_df = pd.DataFrame()
    product_df['Description'] = df['Description']

    stop_words = set(stopwords.words('english'))
    product_df['Clean_Description'] = product_df['Description'].apply(lambda x: ' '.join([word.lower() for word in word_tokenize(x)
                                                                                          if word.lower() not in stop_words
                                                                                          and not word.isdigit()]))

    word_counts = Counter(' '.join(product_df['Clean_Description']).split())
    top_words = word_counts.most_common(40)

    word_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
    plt.figure(figsize=(12, 6))
    plt.bar(word_df['Word'], word_df['Count'])
    plt.xticks(rotation=90)
    plt.xlabel('Word')
    plt.ylabel('Count')
    plt.title('Top 40 Words')
    plt.tight_layout()
    plt.show()





Yanlışlıkla da olsa, Fransa için tüm ürünleri incelerken, bazı ürünlerin yanında POSTAGE---
diye bir ürün de alınıyormuş. Burada şu anlaşıyorki, online alınan ürünleri de çıkarmış oluyoruz.
Tabii bunlar bir anlamda da ürünleri incelememize karşı çıkıyor. Bu yüzden POSTAGE satırlarını---
SUPPORT ve CONFIDENCE hesaplarken dikkate almadan hesaplıyoruz. Bu işlemde aslında güzel analizler---
çıkabilir ama şuan için çok irdelemiyorum.

Veri setindeki aykırı değerleri de temizlik işlemine sokuyoruz. Çünkü temizleme işleminde tüm parametleri---
sağlayıp arada 850 sterlin civarında garip garip ücretler çıkıyor. Bunların olasılık-istatistiklik bilgi---
bilgisiyle, iki fonksiyon yardımı kullanarak belirleyip, temizliyoruz. 
"""
