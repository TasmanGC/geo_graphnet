import numpy as np
import sklearn.metrics as skm
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import minkowski
import csv
from scipy.spatial import distance_matrix
from sklearn.preprocessing import minmax_scale

def mink_calc(node_set,file_name):
    for i,n in enumerate(node_set):
        n.params['iD'] = i

    vectors = np.array([np.array([n.nloc[0],n.nloc[1],n.nloc[2], n.params['cond'],n.params['runc']]) for n in node_set])
    
    with open(file_name, "w",newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i,v in tqdm(enumerate(vectors)):
            a = distance_matrix(vectors[i].reshape(1,-1),vectors, p=1)
            writer.writerow(a[0])

def mink_similarity_2(mink_path,node_list,n_cons=2,file_name=None):
    # lets know all our lines yeah
    lines = list(set([x.params['Line'] for x in node_list]))

    # a surprise tool that'll help us later
    for i in range(len(node_list)):
        node_list[i].params['ID_MAN'] = i

    # we wanna define our sparse connections
    u = []
    v = []

    with open(mink_path, newline='') as f:
        reader = csv.reader(f)

        for i, row in tqdm(enumerate(reader)):
            # should be same number of rows as there are nodes
            i_node = node_list[i]

            ## list comprehensions are used to select all but the current node
            # the minkowski distance to other nodes
            i_mdist = np.array([x for j,x in enumerate(row) if j!=i])
            # the line of all other nodes
            i_line = np.array([x.params['Line'] for j,x in enumerate(node_list) if j!=i])
            # the correct node ID of other nodes
            i_onode = np.array([x.params['ID_MAN'] for j,x in enumerate(node_list) if j!=i])

            # connect here to the other lines
            i_v = []
            for l in lines:
                # lets filter some things by lines 
                cline_dis = i_mdist[np.where(i_line==l)]
                cline_ids = i_onode[np.where(i_line==l)]
                
                idx = np.argpartition(cline_dis,n_cons)
                ids = cline_ids[idx[:n_cons]]
                i_v.extend(ids)
            i_u = [i_node.params['ID_MAN']]*len(i_v)

            # extend our overall u-v
            u.extend(i_u)
            v.extend(i_v)

    if file_name is None:
        file_name = 'mink_sim'+str(len(lines))+'.csv'
            
    graph_edge_df = {}
    graph_edge_df['U'] = u 
    graph_edge_df['V'] = v 
    edges = pd.DataFrame.from_dict(graph_edge_df)
    edges.to_csv(file_name)


def node_sim(node_a,node_b):

    v_1     = [val for _list in [node_a.nloc, [node_a.params['cond']],[node_a.params['runc']]] for val in _list]
    v_2     = [val for _list in [node_b.nloc, [node_b.params['cond']],[node_b.params['runc']]] for val in _list]
        #calculate minkowski distance
    m_dist  = minkowski(v_1,v_2,1)
    return(m_dist)

def edge_weight(n_set,e_set):
    '''
    e_set dataframe should have only u and v columns
    '''
    m_dist_all = []

    for u,v in e_set.values:
        v_1     = [val for _list in [n_set[u].nloc, [n_set[u].params['cond']],[n_set[u].params['runc']]] for val in _list]
        v_2     = [val for _list in [n_set[v].nloc, [n_set[v].params['cond']],[n_set[v].params['runc']]] for val in _list]
        #calculate minkowski distance
        m_dist  = minkowski(v_1,v_2,1)
        m_dist_all.append(m_dist)
    
    # lets convert distance to an edge weights
    norm        = minmax_scale(m_dist_all)
    e_weights   = [1-n for n in norm]

    return(e_weights)

def calculate_lattice_conections(n_set,max_cross=1):
    for i,n in enumerate(n_set):
        n.params['iD'] = i

    line_list = np.unique([node.params["Line"] for node in n_set])

    # some useful parameters in our dataset
    node_per_line   = {x : len([y for y in n_set if y.params['Line']==x]) for x in line_list}
    line_stations   = {x : sorted(list(set([(y.nloc[0],y.nloc[1]) for y in n_set if y.params['Line']==x]))) for x in line_list}
    
    # lets sort our nodes 
    sorted_n_set = []
    for line in line_list:
        line_nodes = [x for x in n_set if x.params['Line']==line]
        line_nodes.sort(key=lambda x: (x.nloc[0], x.nloc[2]))
        sorted_n_set.extend(line_nodes)

    # caclulate connections between adjacent lines
    node_l = list(node_per_line.values())
    result = [0] + list(np.cumsum(node_l))

    # use these to make cool stuff
    x = np.array([x.nloc[0] for x in n_set])
    y = np.array([x.nloc[1] for x in n_set])
    z = np.array([x.nloc[2] for x in n_set])
    spatial = np.array(list(zip(x,y,z)))

    # top and bottom
    top = np.arange(0,len(n_set),30,dtype=int)
    bot = np.arange(29,len(n_set),30,dtype=int)

    u=[]
    v=[]

    for i, node in enumerate(tqdm(sorted_n_set)):

        # top or botoom
        t_ = True if i in top else False
        b_ = True if i in bot else False

        # define the cons
        if t_:
            u.extend([i])
            v.extend([i+1])
        if b_:
            u.extend([i])
            v.extend([i-1])
        if not t_ and not b_:
            u.extend([i]*2)
            v.extend([i+1,i-1])

        # this station and next station
        frst = True if line_stations[node.params['Line']][0]==(node.nloc[0],node.nloc[1]) else False
        last = True if line_stations[node.params['Line']][-1] == (node.nloc[0],node.nloc[1]) else False
        
        # defining cons
        if frst:
            u.extend([i])
            v.extend([i+30])

        if last:
            u.extend([i])
            v.extend([i-30])

        if not frst and not last:
            u.extend([i]*2)
            v.extend([i+30,i-30])

        frst_line = True if list(line_stations)[0]==node.params['Line'] else False
        last_line = True if list(line_stations)[-1]==node.params['Line'] else False

        if frst_line: # node is in first line
            n_line = list(line_stations)[list(line_stations).index(node.params['Line'])+1]
            n_nodes = [n for n in n_set if n.params['Line']==n_line]
            n_spati = np.array([x.nloc for x in n_nodes])
            n_ids   = np.array([x.params['iD'] for x in n_nodes])
            dist = skm.pairwise.euclidean_distances(X=n_spati, Y=np.array(node.nloc).reshape(1,-1)).flatten()
            n_closest = np.argpartition(dist,max_cross)[:max_cross]
            u.extend([node.params['iD']]*max_cross)
            v.extend(n_ids[n_closest])

        if last_line: # node is in last line
            p_line = list(line_stations)[list(line_stations).index(node.params['Line'])-1]
            p_nodes = [n for n in n_set if n.params['Line']==p_line]
            p_spati = np.array([x.nloc for x in p_nodes])
            p_ids   = np.array([x.params['iD'] for x in p_nodes])
            dist = skm.pairwise.euclidean_distances(X=p_spati, Y=np.array(node.nloc).reshape(1,-1)).flatten()
            p_closest = np.argpartition(dist,max_cross)[:max_cross]
            u.extend([node.params['iD']]*max_cross)
            v.extend(p_ids[p_closest])

        if not frst_line and not last_line: # node has two adjacent lines
            p_line = list(line_stations)[list(line_stations).index(node.params['Line'])-1]
            p_nodes = [n for n in n_set if n.params['Line']==p_line]
            p_spati = np.array([x.nloc for x in p_nodes])
            p_ids   = np.array([x.params['iD'] for x in p_nodes])
            dist = skm.pairwise.euclidean_distances(X=p_spati, Y=np.array(node.nloc).reshape(1,-1)).flatten()
            p_closest = np.argpartition(dist,max_cross)[:max_cross]
            u.extend([node.params['iD']]*max_cross)
            v.extend(p_ids[p_closest])

            n_line = list(line_stations)[list(line_stations).index(node.params['Line'])+1]
            n_nodes = [n for n in n_set if n.params['Line']==n_line]
            n_spati = np.array([x.nloc for x in n_nodes])
            n_ids   = np.array([x.params['iD'] for x in n_nodes])
            dist = skm.pairwise.euclidean_distances(X=n_spati, Y=np.array(node.nloc).reshape(1,-1)).flatten()
            n_closest = np.argpartition(dist,max_cross)[:max_cross]
            u.extend([node.params['iD']]*max_cross)
            v.extend(n_ids[n_closest])
    return(u,v)


def n_max(arr, n):
    indices = arr.ravel().argsort()[-n:]
    indices = (np.unravel_index(i, arr.shape) for i in indices)
    return([(arr[i], i) for i in indices])