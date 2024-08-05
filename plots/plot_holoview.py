import os
import numpy as np
import param
import panel as pn
import holoviews as hv
from . import reconstruction_plots as recplot
from bokeh.models import HoverTool


hv.extension('bokeh')
hv.opts.defaults(hv.opts.ErrorBars(width=300, height=200, lower_head=None, 
                        upper_head=None, xrotation=35, line_width = 0.8, alpha = 0.8),
                 hv.opts.Scatter(width=300, height=200, size = 3, alpha = 0.8),
                 hv.opts.Curve(width=300, height=200, line_dash = 'dashed',
                        line_width=.3, alpha = 1.),
                 hv.opts.Area(width=300, height=200,  alpha = 0.05) )

def save_all_curves(metrics_ext, metrics_plot, folder_root, config, saved_name = '',
             is_scaled = False, is_obs_data = True, count = 50):
  this_count = 0
  for i in range(len(metrics_ext['id'])):
    recplot.single_curve(i, metrics_ext, metrics_plot, config, folder_root, saved_name = saved_name,
                            is_scaled = False, is_obs_data = is_obs_data)
    this_count += 1
    if this_count > count:
      break

def create_plot_aux(list_scores, list_target, classes_names, config):
  global alist_scores
  global alist_target
  global aclasses_names
  global page_number
  page_number  = 3
  alist_scores = list_scores
  alist_target = list_target 
  aclasses_names = classes_names
  class PlotExplorer(param.Parameterized):
    page                = param.Integer(default=0, bounds=(0, page_number - 1))
    scores              = param.ObjectSelector(default=alist_scores[0], objects = alist_scores)
    target_distribution = param.ObjectSelector(default=alist_target[0], objects = alist_target)
    class_selected      = param.ObjectSelector(default= aclasses_names[0], objects = aclasses_names.tolist())
    @param.depends('scores', 'target_distribution', 'class_selected', 'page')
    def __init__(self, list_scores, list_target, config, is_scaled = False, is_obs_data = True):
      super().__init__()
      self.list_scores    = list_scores
      self.list_target    = list_target
      self.classes_names  = config['classes_names'].tolist()
      self.page_number    = page_number
      self.lc_per_page    = 12
      self.config         = config
      self.is_obs_data    = is_obs_data
      self.ss             = '' if not is_scaled else 'scaled_'
    def plot_lc(self, metrics_ext, metrics_plot, index, page):
        plot_bands = []
        ss     = self.ss
        config = self.config
        for cc in range(config['dataset_channel']):
            if not self.is_obs_data:
              mask_cc = np.arange(len(metrics_plot['time'][index, :, cc]))
            else:
              mask_cc = metrics_plot['mask'][index, :, cc].astype(bool)
            if self.is_obs_data:
              plot_bands.append(hv.Scatter((metrics_plot['time'][index, :, cc][mask_cc],
                  metrics_plot[ss+'data'][index,:, cc][mask_cc])).opts(line_color=config['band_colors'][cc], marker = 'o'))
              if config['noise_data']:
                plot_bands.append(hv.ErrorBars((metrics_plot['time'][index, :, cc][mask_cc],
                  metrics_plot[ss+'data'][index,:, cc][mask_cc],
                  metrics_plot[ss+'data_sigma'][index,:, cc][mask_cc])).opts(line_color=config['band_colors'][cc]))
            plot_bands.append(hv.Curve((metrics_plot['time'][index, :, cc][mask_cc],
                  metrics_plot[ss+'D_mu'][index, :, cc][mask_cc])).opts(line_color = config['band_colors'][cc] ))
            if config['is_dec_var']:
              plot_bands.append(hv.Area((metrics_plot['time'][index, :, cc][mask_cc],
                metrics_plot[ss+'D_mu'][index,:, cc][mask_cc] - metrics_plot[ss+'D_sigma'][index,:, cc][mask_cc],
                metrics_plot[ss+'D_mu'][index,:, cc][mask_cc] + metrics_plot[ss+'D_sigma'][index,:, cc][mask_cc]),
                vdims=['y1', 'y2']).opts(line_color=config['band_colors'][cc]))
            if config['which_post_decoder'] != '':
              plot_bands.append(hv.Scatter((metrics_plot['time_prot'][index, :, cc],
                  metrics_plot[ss+'D_prot_x'][index,:, cc])).opts(line_color=config['band_colors'][cc], marker = 'o'))
              if config['is_dec_var']:
                plot_bands.append(hv.ErrorBars((metrics_plot['time'][index, :, cc][mask_cc],
                  metrics_plot[ss+'D_prot_x'][index,:, cc][mask_cc],
                  np.sqrt(metrics_plot[ss+'D_prot_x_var'][index,:, cc]) )).opts(line_color=config['band_colors'][cc])  )
        return hv.Overlay(plot_bands).opts(shared_axes=False, framewise=True, title = str(metrics_ext['id'][index]) + ' __ ' + str(page),
                                          fontsize={'labels': 0, 'title':8,  'legend': 1, 'legend_title':0}, legend_position='bottom_right')
    def create_grid(self, metrics_ext, metrics_plot):
      #if self.target_distribution == 'None' or self.target_distribution is None:
      #  valid_index = np.arange(len(metrics_ext['y']))
      #else:
      class_index = self.classes_names.index(self.class_selected)
      valid_index = metrics_ext[self.target_distribution] == class_index
      ar          = np.arange(len(metrics_ext['y']))
      scores_used = metrics_ext[self.scores][valid_index]
      ar_index    = ar[valid_index]
      arg_scores  = scores_used.argsort()
      ar_index    = ar_index[arg_scores]
      ar_index    = ar_index[np.floor(np.linspace(0, len(ar_index) - 1, self.lc_per_page*self.page_number)).astype(int)]
      id_selected = ar_index[(self.page)*self.lc_per_page:(self.page+1)*self.lc_per_page]
      return hv.Layout([self.plot_lc(metrics_ext, metrics_plot, oid, self.page) for oid in id_selected ]).cols(self.lc_per_page//3).opts(framewise=True, shared_axes=False)
    def view(self, metrics_ext, metrics_plot):
      return hv.DynamicMap(self.create_grid(metrics_ext, metrics_plot) ).opts(shared_axes=False, framewise=True)      
  return PlotExplorer(list_scores, list_target, config, is_scaled = False, is_obs_data = True)
def create_plot(metrics_ext, metrics_plot, list_scores, list_target, config):
  explorer = create_plot_aux(list_scores, list_target, config['classes_names'], config)
  #explorer = PlotExplorer(list_scores, list_target, config, is_scaled = is_scaled, is_obs_data = True)
  app = pn.Column(explorer.view(metrics_ext, metrics_plot), explorer.param)
  # If on jupyter you can run app to display the dashboard
  app.save("%s/grid.html" % config['holoview_root'], embed=True)

def create_scatter_aux(list_scores, list_target, classes_names, config):
  global alist_scores
  global alist_target
  global aclasses_names
  alist_scores = list_scores
  alist_target = list_target 
  aclasses_names = classes_names

  class ScatterExplorer(param.Parameterized):
      scores              = param.ObjectSelector(default=alist_scores[0], objects = alist_scores)
      target_distribution = param.ObjectSelector(default=alist_target[0], objects = alist_target)
      class_selected      = param.ObjectSelector(default=aclasses_names[0], objects = aclasses_names.tolist())
      @param.depends('scores', 'target_distribution', 'class_selected')
      def __init__(self, list_scores, list_target, config):
        super().__init__()
        self.list_scores    = list_scores
        self.list_target    = list_target
        self.classes_names  = config['classes_names'].tolist()
      def obtain_scan(self, key_name, key):
        aux = """
                  <div>
                    <span style="font-size: 16px; color: #696;">(%s, %s)</span>
                  </div>
                """ % (key_name, "@%s" % key)
        return aux
      def search_list(self, key_list):
        aux = ""
        for key in key_list:
          aux += self.obtain_scan(key, key)
        return aux
      def obtain_TOOLTIPS(self):
        TOOLTIPS      = ""
        TOOLTIPS     += " <div>"
        TOOLTIPS     += """
                          <div>
                              <img
                                  src="@imgs" height="450" alt="@imgs" width="450"
                                  style="float: center; margin: 0px 0px 0px 0px;"
                                  border="2"
                              ></img>
                          </div> 
                          """
        TOOLTIPS     += self.search_list(self.list_target)
        TOOLTIPS     += self.search_list(self.list_scores)
        TOOLTIPS     += "</div>"
        return TOOLTIPS
      def view(self, pd_data):
        #return hv.DynamicMap(self.create_grid).opts(shared_axes=False, framewise=True)}
        if self.target_distribution == 'None':
          pd_data_aux = pd_data
          used_name   = 'y'
        else:
          class_index = self.classes_names.index(self.class_selected)
          pd_data_aux = pd_data[pd_data[self.target_distribution] == class_index]
        html_dep    = ["imgs"] + self.list_scores + self.list_target
        hover_tool = HoverTool(tooltips = self.obtain_TOOLTIPS())
        import pdb
        pdb.set_trace()

        h1 = hv.Points(pd_data_aux, ["z1", "z2"], html_dep).opts(tools=[hover_tool],
                                  alpha=0.5, size = 16, width = 800, height = 700, color = self.target_distribution)
        h2 = hv.Points(pd_data_aux, ["z1", "z2"], html_dep).opts(tools=[hover_tool],
                                  alpha=0.5, size = 16, width = 800, height = 700, color = self.scores, colorbar = True)
        return pn.Row(h1,h2)
      #return h2
  return ScatterExplorer(list_scores, list_target, config)
def create_scatter(pd_data, list_scores, list_target, config, name_scatter = ''):
  #explorer = ScatterExplorer(list_scores, list_target, config)
  explorer = create_scatter_aux(list_scores, list_target, config['classes_names'], config)
  app      = pn.Column(explorer.view(pd_data), explorer.param)
  app.save("%s/scatter%s.html" % (config['holoview_root'], name_scatter), embed=True)