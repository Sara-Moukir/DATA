import mpi4py.rc
mpi4py.rc.threads = False
from mpi4py import MPI

import os
import lxml.etree as et
import subprocess
import gzip
import shutil
import sys
import numpy as np
import functions as f
#import init as i


#i.reset()

# Number of agents in the scenario
number_of_agents = 19123



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print("size :", size)

value_first = 0
value_last = 2

def unzip(path):
    with gzip.open(path) as f_in:
        with open('/gpfs/users/moukirs/MATSIMLA/matsim' + str(rank) + '/matsim-los-angeles/scenarios/los-angeles-v1.1/output/los-angeles-v1.1-0.1pct_run0/los-angeles-v1.1-0.1pct_run0.output_plans.xml', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)


def zip(path):
    with open(path, 'rb') as f_in:
        with gzip.open('/gpfs/users/moukirs/MATSIMLA/matsim' + str(rank) + '/matsim-los-angeles/scenarios/los-angeles-v1.1/output/los-angeles-v1.1-0.1pct_run0/los-angeles-v1.1-0.1pct_run0.output_plans.xml.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)



for x in range(4): 

    tab_of_scores = np.empty(number_of_agents, dtype = np.float64)


    # Execute MATSim from iter x to x+10
    bashCmd = ['java -Xmx50000m -cp /gpfs/users/moukirs/MATSIMLA/matsim' + str(rank) + '/matsim-los-angeles/matsim-los-angeles-v1.1-jar-with-dependencies.jar org.matsim.run.RunLosAngelesScenario /gpfs/users/moukirs/MATSIMLA/matsim' + str(rank) + '/matsim-los-angeles/scenarios/los-angeles-v1.1/input/los-angeles-v1.1-0.1pct.config.xml']
    process = subprocess.Popen(bashCmd, shell = True)
    output, error = process.communicate()
    print(rank, "bash command executed")
        # Open and parse xml file
    tree = et.parse('/gpfs/users/moukirs/MATSIMLA/matsim' + str(rank) + '/matsim-los-angeles/scenarios/los-angeles-v1.1/input/los-angeles-v1.1-0.1pct.config.xml')
    root = tree.getroot()
    print(rank, "config file parsed")
    # Define first and last iteration in xml file
    value_first = value_last 
    value_last = value_first + 2
        
    for first in tree.xpath("/config/module[@name='controler']/param[@name='firstIteration']"):
        first.attrib['value'] = str(value_first)
            
    for last in tree.xpath("/config/module[@name='controler']/param[@name='lastIteration']"):
        last.attrib['value'] = str(value_last)
            
    for plan in tree.xpath("/config/module[@name='plans']/param[@name='inputPlansFile']"):
        plan.attrib['value'] = '/gpfs/users/moukirs/MATSIMLA/matsim' + str(rank) + '/matsim-los-angeles/scenarios/los-angeles-v1.1/output/los-angeles-v1.1-0.1pct_run0/los-angeles-v1.1-0.1pct_run0.output_plans.xml.gz'
            
    
    tree.write('/gpfs/users/moukirs/MATSIMLA/matsim' + str(rank) + '/matsim-los-angeles/scenarios/los-angeles-v1.1/input/los-angeles-v1.1-0.1pct.config.xml', xml_declaration=True, encoding='UTF-8')
    print(rank, "values in xml file changed")  
    # Unzip file to extract scores from output_plans.xml
    p = '/gpfs/users/moukirs/MATSIMLA/matsim' + str(rank) + '/matsim-los-angeles/scenarios/los-angeles-v1.1/output/los-angeles-v1.1-0.1pct_run0/los-angeles-v1.1-0.1pct_run0.output_plans.xml.gz'
    unzip(p)

    unzipped_p = '/gpfs/users/moukirs/MATSIMLA/matsim' + str(rank) + '/matsim-los-angeles/scenarios/los-angeles-v1.1/output/los-angeles-v1.1-0.1pct_run0/los-angeles-v1.1-0.1pct_run0.output_plans.xml'


    # Parse and explore file to extract scores and put it in a vector 
    tree2 = et.parse('/gpfs/users/moukirs/MATSIMLA/matsim' + str(rank) + '/matsim-los-angeles/scenarios/los-angeles-v1.1/output/los-angeles-v1.1-0.1pct_run0/los-angeles-v1.1-0.1pct_run0.output_plans.xml.gz').getroot()
    print(rank, "output plan parsed")    
    for ids in tree2.findall("./person"):
        id = ids.attrib['id']
        for score in ids.findall("plan/[@selected = 'yes']"):
            scr = score.attrib['score']
            tab_of_scores[int(id)-1] = float(scr)
        
    print(rank, "scores recolted from outputplans")
    
    print('***************************************************************','iter =', x, 'rank =', rank, 'value first =', value_first, 'value_last =', value_last)


    
    # At this step, we have all the scores of the selected plans for all agents in tab_of_scores    

###################### GATHERING SCORES OF OTHER RANKS ##################

    # buff will countain all the scores of the selected plans for all the agents AND all the ranks
    buff = np.empty(number_of_agents*size, dtype = np.float64)
    print(rank, "length of buff nb_ag*size: ", len(buff))
    print(rank, "communication 1 begins...")
    # Communication between ranks for scores
    comm.Allgather([tab_of_scores, MPI.FLOAT], [buff, MPI.FLOAT])
    print(rank, "communications of scores between processes OK")

########### Sending the scenario files to others and gathering the ones of the other ranks ###############
    
    # Gathering first the length in bytes for each files 
    # Convert the output plans file into bytes to send it to other ranks
    src_byt = et.tostring(tree2, encoding = 'utf-8')



    # Buffer for receiving lengths
    ll = len(src_byt)
    
    local_length = np.array([ll])
  
    length_scenario_files_table = np.empty(size, dtype = int)
    print(rank, "communication 2 begins...")
    # Communication of lengths
    comm.Allgather([local_length, MPI.INT], [length_scenario_files_table, MPI.INT])
    print(rank, "length_scenario_files_table:", length_scenario_files_table)
    print(rank, "communication of lengths OK")
    # Buffer for receiving files
    files_buffer = bytearray(sum(length_scenario_files_table))

    print("sum of files_buffer :", sum(length_scenario_files_table) )
    print(rank, "communication 3 begins...")

    # Communication of files 
    comm.Allgatherv([src_byt, MPI.BYTE], [files_buffer, length_scenario_files_table , MPI.BYTE])
    print(rank, "communication of files OK")
     # Now it's time to "parse" the files_buffer to separate all the "size" files
    delimiter = b"</population>"

    output_files = [yy+delimiter for yy in files_buffer.split(delimiter) if yy]



    # if rank == 0:

    #     sourceFile = open('yo.txt', 'w')
    #     print(output_files[0], file = sourceFile)
    #     sourceFile.close()



    
    ################## COMPUTING THE BEST SCORES AMONG ALL THE RANKS FOR EACH AGENT AND CREATING THE VECTORS OF BEST SCORES, ASSOCIATED RANK ANS STD ###########################
    #buf_poc = np.random.randint(50, size = size * number_of_agents)
    
    #vect_rank_best_scores, vect_best_scores, vect_std, matt = l.largestInColumn(buff, 4, 100) buf_poc est un tableau random pour faire les tests, puisque les scenario
    #sont tous les mêmes
    
    print("finding the max scores for each agent in each rank starts...")
    
    vect_rank_best_scores, vect_best_scores, vect_std, matt = f.largestInColumn(buff, size, number_of_agents) 

    #Comparison between the abs of best scores minus the score of a rank, and test if it's greater than the std or not
    bool_vec = np.greater(abs(tab_of_scores - vect_best_scores), vect_std)

    print("length bool_vec =", len(bool_vec))
    #for xx in range(len(bool_vec)):
     #   if bool_vec[xx]:
        
            #print("rank", str(rank), ":", "agent", str(xx+1), "needs plan of rank", str(vect_rank_best_scores[xx]))
            
      #      f.xml_extractor(output_files[vect_rank_best_scores[xx]], unzipped_p, xx+1)

    #zip(unzipped_p)




    




    # print("vect best scores", vect_best_scores)
    # print("\n")

    # print("matrix", matt)
    # print("\n")


    # print("vect rank", vect_rank_best_scores)
    # print("\n")


    # print("vect std", vect_std)
    # print("\n")
