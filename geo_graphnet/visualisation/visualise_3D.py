import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import dgl
from typing import Tuple, List


class Scene3D:
    def __init__(
        self,
        graph: dgl.graph,
        parameter: str,
        loc_keys: List = ["x_loc", "y_loc", "z_loc"],
        plot_title: str = "Graph Data",
    ):
        self.graph = graph
        self.loc_keys = loc_keys
        self.parameter_key = parameter
        self.plot_title = plot_title
        self.pv_graph = None
        self.pv_edges = None

        self.sargs = dict(
            height=0.1,
            width=0.50,
            vertical=False,
            position_x=0.25,
            position_y=0.1,
            title=plot_title,
        )

        pass

    def _create3DGraph(self):
        xp = np.array(self.graph.ndata[self.loc_keys[0]])
        yp = np.array(self.graph.ndata[self.loc_keys[1]])
        zp = np.array(self.graph.ndata[self.loc_keys[2]])
        pa = np.array(self.graph.ndata[self.parameter_key])

        vert = list(zip(xp, yp, zp))
        pv_cloud = pv.PolyData(vert)
        pv_cloud[self.parameter_key] = pa

        self.pv_graph = pv_cloud

    def _create3DEdges(self):
        xp = np.array(self.graph.ndata[self.loc_keys[0]]).tolist()
        yp = np.array(self.graph.ndata[self.loc_keys[1]]).tolist()
        zp = np.array(self.graph.ndata[self.loc_keys[2]]).tolist()
        points = list(zip(xp, yp, zp))

        edges = list(zip([x.detach().numpy().tolist() for x in self.graph.edges()]))
        edges = list(zip(edges[0][0], edges[1][0]))
        edges = np.hstack([[2, x[0], x[1]] for x in edges])
        edges = pv.PolyData(points, lines=edges, n_lines=self.graph.num_edges())
        self.pv_edges = edges

    def interactive(
        self,
        win_size: Tuple = (1500, 1000),
        plot_cons: bool = False,
        cpos=None,
        c_map: str = "GnBu_r",
        ps: int = 5,
        sbar_args: dict = {},
    ):
        # update class state
        self.plot_cons = plot_cons
        self.cpos = cpos

        plotter = pv.Plotter(notebook=False, title=self.plot_title)

        if self.cpos != None:
            plotter.camera_position = cpos

        plotter.window_size = win_size

        # create polydata_objects
        self._create3DGraph()

        if self.plot_cons:
            self._create3DEdges()

        plotter.add_mesh(
            self.pv_graph,
            render_points_as_spheres=True,
            cmap=c_map,
            point_size=ps,
            scalar_bar_args=sbar_args,
        )

        if self.plot_cons:
            plotter.add_mesh(
                self.pv_edges, show_edges=True, edge_color=[0, 0, 0], opacity=0.2
            )

        # visualise and save
        cpos = plotter.show(return_cpos = True)
        return cpos
    
    def save_image(self,
                   save_name = "default.png",
                   win_size: Tuple = (1500, 1000),
                   plot_cons: bool = False,
                   cpos=None,
                   c_map: str = "GnBu_r",
                   ps: int = 50,
                   sbar_args: dict = {},
                ):
        # update class state
        self.plot_cons = plot_cons
        self.cpos = cpos

        plotter = pv.Plotter(off_screen=True, title=self.plot_title)
        #plotter.off_screen = True

        if self.cpos != None:
            plotter.camera_position = cpos
        
        plotter.window_size = win_size

        # create polydata_objects
        self._create3DGraph()

        if self.plot_cons:
            self._create3DEdges()

        plotter.add_mesh(
            self.pv_graph,
            render_points_as_spheres=True,
            cmap=c_map,
            point_size=ps,
            scalar_bar_args=sbar_args,
        )

        if self.plot_cons:
            plotter.add_mesh(
                self.pv_edges, show_edges=True, edge_color=[0, 0, 0], opacity=0.2
            )
            
        plotter.show(auto_close=False)
        image = plotter.screenshot(save_name, scale=4)

    # def animation(self,cpos=None): # FUTURES
    #     pass

