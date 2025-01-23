def txt_to_dataframe(path):
    dfdict2 = {}
    print(path)
    with open(path,'r') as file:
        for line in file.readlines():
            line = line.split()
            line.remove('Timestamp:')
            line.remove('ID:')
            line.remove('DLC:')
            # if len(line) > 4:
            #     line[4] = line[4:]
            #     line = line[:5]            
            dfdict2[float(line[0])] = line[1:]
            # print(line)

    #create df from dictionary
    df2 = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in dfdict2.items() ]))
    df2 = df2.T
    df2 = df2.iloc[:,:-1]
    df2

    #set index and column names
    # df2.set_index('Timestamp')
    df2.columns = ['CAN ID','RTR','DLC','Data1','Data2','Data3','Data4','Data5','Data6','Data7','Data8'] 
    df2.index = df2.index.rename('Timestamp')


    # replace NaNs with 0s
    df2 = df2.fillna('0')
    #convert hex strings and objects to ints
    #change datatypes from hex to int
    df2['CAN ID'] = df2['CAN ID'].apply(lambda x: int(x, 16))
    df2['DLC'] = df2['DLC'].astype('int32')
    df2['RTR'] = df2['RTR'].astype('int32')
    b16 = lambda x: int(x,16)
    for col in ['Data1','Data2','Data3','Data4','Data5','Data6','Data7','Data8']:
        df2[col] = df2[col].apply(b16)
    
    return df2