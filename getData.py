import json
import os
import zipfile
import numpy as np
import sys
import logging

logger = None

def get_dataload(result_path,result_name,num:str,sex:str,age:str,dataloc:str):
    
    file_path = '/home/quve/Desktop/open_data/opendata'
    file_names = os.listdir(file_path)
    num = int(num)
    if num > 100:
        num = 100
    if dataloc == 'U':
        uplow = 0
    elif dataloc == 'L':
        uplow = 1
    elif dataloc == 'A':
        uplow = 2
    
    jsonlist = []
    objlist= []
    for file_name in file_names:
        
        newfilename = file_name.replace('.','_')
        pof = newfilename.split("_")
            
        if file_name.endswith('obj'):
            if sex == "A" and dataloc == "A" and age == "0":
                objlist.append(file_name)
            
            if sex == "A":
                if uplow < 2 and int(pof[0]) % 2 != uplow:
                    continue
                if pof[1][0] != age[0]:
                    continue
                objlist.append(file_name)

            if dataloc == "A":
                if pof[1][0] != age[0]:
                    continue
                if pof[2] != sex:
                    continue
                objlist.append(file_name)

            if age == "0":
                if uplow < 2 and int(pof[0]) % 2 != uplow:
                    continue
                if pof[2] != sex:
                    continue
                objlist.append(file_name)
            
            if sex != "A" and dataloc != "A" and age != "0": 
                if uplow < 2 and int(pof[0]) % 2 != uplow:
                    continue
                if pof[1][0] != age[0]:
                    continue
                if pof[2] != sex:
                    continue
                objlist.append(file_name)

    if len(objlist) < num:
        objlist = objlist
    elif len(objlist) >= num:
        objlist = np.random.choice(objlist, num).tolist()

    for file in objlist:
        jsonlist.append(file[:-3]+"json")
    finallist = objlist + jsonlist

    curDir = os.getcwd()
    os.chdir(file_path)
    
    zip = zipfile.ZipFile(os.path.join(result_path, result_name), 'w')
    for finalfile in finallist:
        zip.write(finalfile, compress_type = zipfile.ZIP_DEFLATED)
    zip.close()

    os.chdir(curDir)

def main(result_path,result_name,num,sex,age,dataloc):
    # Logger
    global logger
    logging.basicConfig(filename="getData.log", filemode="a+", format="[%(asctime)s][%(levelname)s] - %(message)s", level=logging.INFO)
    logger = logging.getLogger()
    
    log("START main - {} {} {} {} {} {} ".format(result_path, result_name, num, sex, age,dataloc))
    try:    
        get_dataload(result_path,result_name,num,sex,age,dataloc)
    except Exception as e:
        log(e.__str__(), True)
        
    log("FINISHED main")
def log(logMessage, exc_info = False):
    global logger
    
    if exc_info :
        logger.error(logMessage, exc_info=True)
    else:
        logger.info(logMessage)
    
if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6])