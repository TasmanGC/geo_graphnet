import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


def visualise_3D(graph, parameter,cpos, cons=None, con_val = None, sbar_args=None, meshes=None, c_map='GnBu_r',  auto_close=False, win_size=(1500,1000) , plot_title='AEM Data',bg_color=None,save_name = None, clip=None, heme=None, ps=5):
    """ Plots provided data in 3D using pyvista.
    Args:
        node_list (list)    : List of node objects.
        parameter (str)     : Dictionary key for node object params attribute.

        cons (list, optional)   : List of Polydata objects. Defaults to None.
        meshes (list, optional) : List of Polydata objects. Defaults to None.

    Raises:
        Window: This function raises a visualisation window.

    """

    xp = np.array(graph.ndata['x_loc'])                                    # 0.0 - collect the x of the points
    yp = np.array(graph.ndata['y_loc'])                                    # 0.1 - collect the y of the points
    zp = np.array(graph.ndata['z_loc'])                            # 0.2 - collect the z of the points
    pa = [node.params[parameter] for node in graph.ndata[parameter]] 

    vert = list(zip(xp,yp,zp))

    pv_cloud = pv.PolyData(vert)                       # 1.0 - put the co-ords together
    pv_cloud[parameter] = pa                                                    # 1.1 - collect the 

    sargs = dict(height=0.1,width=0.50, vertical=False, position_x=0.25, position_y=0.1, title=plot_title)
    if sbar_args!=None:
        sargs = {**sargs,**sbar_args}
    
    # generate our plotter
    if auto_close:
        plotter = pv.Plotter(notebook=False,title=plot_title,off_screen=auto_close)
    if not auto_close:
        plotter = pv.Plotter(notebook=False,title=plot_title)



    plotter.camera_position = cpos
    plotter.store_image = True
    plotter.window_size = win_size
    
    #TODO update to work based on a graph object
    # # populate our plotter 
    # if cons == True:
    #     # working 3D visualisation
    #     points  = [x.nloc for x in self.nodes]
    #     edges   = list(zip(list(self.conns['U'].values),list(self.conns['V'].values)))
    #     edges   = np.hstack([[2,x[0],x[1]] for x in edges])
    #     edges   = pv.PolyData(points,lines=edges,n_lines=len(self.conns['U'].values))

    #     if con_val!=None:
    #         plotter.add_mesh(edges,show_edges=True,edge_color=[0,0,0])
    #     if con_val==None:
    #         plotter.add_mesh(edges,show_edges=True,edge_color=[0,0,0],opacity=0.2)
    
    if clip == None:
        plotter.add_mesh(pv_cloud,render_points_as_spheres=True, cmap=c_map,point_size=ps,scalar_bar_args=sargs) 
    
    if clip != None:
        plotter.add_mesh(pv_cloud,render_points_as_spheres=True, cmap=c_map,point_size=ps,scalar_bar_args=sargs,clim=clip) 

    if meshes is not None:
        for i in range(len(meshes)):
            plotter.add_mesh(meshes[i][0],cmap=meshes[i][1])

    if bg_color!=None:
        plotter.set_background(**bg_color)

    
    # visualise and save
    cpos = plotter.show(auto_close=auto_close)
    if save_name != None:
        image = plotter.image
        fig, ax = plt.subplots(figsize=(win_size[0]/100, win_size[1]/100))
        ax.imshow(image)
        plt.tight_layout()
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        plt.savefig(save_name)
    
    if auto_close:
        plotter.close()
        pv.close_all()

    return(cpos)