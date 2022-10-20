import numpy as np
import lxml.etree as et
import gzip


def largestInColumn(mat, rows, cols):
    
    reshaped_matrix = mat.reshape(rows, cols)
    r = np.empty(cols, dtype = np.int64)
    s = np.empty(cols, dtype = np.float64)
    v = np.empty(cols, dtype = np.float64)
    buf = np.empty(rows, dtype = np.float64)
    
    print("rows =", rows)
    print("cols ", cols)
    for ii in range(cols):
        
        # initialize the maximum element with 0
        #maxm = reshaped_matrix[0][ii]
        #maxm = 0
        maxm = reshaped_matrix[0][ii]
        # run the inner loop for news
        for jj in range(rows):
             
            # check if any elements is greater
            # than the maximum elements
            # of the column and replace it
            #Computing the variance
            buf[jj] = reshaped_matrix[jj][ii]
            if reshaped_matrix[jj][ii] >= maxm:
                maxm = reshaped_matrix[jj][ii]
                r[ii] = jj
                #print(jj)
                s[ii] = maxm
                
        
        v[ii] = np.std(buf)
        #print(buf)

    # r : vector containing the rank of the best score for each agent,
    # s : vector containing the best scores for each agent,
    # v : std of each column, so of each score for one agent computed by a rank
    return r, s, v, reshaped_matrix







def xml_extractor(src_bytes, dest, id):
    treesrc = et.ElementTree(et.fromstring(bytes(src_bytes)))
    treedst = et.parse(dest)

    dest_root = treedst.getroot()

    x_phrase = '/person/[@id="' + str(id) + '"]'

    src_tag = treesrc.find(x_phrase)

    for elem in treedst.xpath('/population'):
        for child in elem:
            if str(child.attrib) == "{'id': '" + str(id) + "'}":

                elem.insert(elem.index(child), src_tag)
                elem.remove(child)

    et.ElementTree(dest_root).write(dest, pretty_print=True, encoding='utf-8', xml_declaration=True)
    






