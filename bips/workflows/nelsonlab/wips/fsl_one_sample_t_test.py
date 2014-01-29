import os
from traits.api import HasTraits, Directory, Bool, Button
import traits.api as traits
from .QA_utils import cluster_image
from ...base import MetaWorkflow, load_config, register_workflow
from ...flexible_datagrabber import Data, DataBase

"""
MetaWorkflow
"""
desc = """
Group Analysis: One Sample T-Test (FSL)
=====================================

"""
mwf = MetaWorkflow()
mwf.uuid = 'f08f0a22ac0511e195e90019b9f22493'
mwf.tags = ['FSL', 'second level', 'one sample T test']

mwf.help = desc


"""
Config
"""

class config(HasTraits):
    uuid = traits.Str(desc="UUID")

    # Directories
    working_dir = Directory(mandatory=True, desc="Location of the Nipype working directory")
    base_dir = Directory(os.path.abspath('.'),mandatory=True, desc='Base directory of data. (Should be subject-independent)')
    sink_dir = Directory(mandatory=True, desc="Location where the BIP will store the results")
    crash_dir = Directory(mandatory=False, desc="Location to store crash files")

    # Execution
    run_using_plugin = Bool(False, usedefault=True, desc="True to run pipeline with plugin, False to run serially")
    plugin = traits.Enum("PBS", "MultiProc", "SGE", "Condor",
                         usedefault=True,
                         desc="plugin to use, if run_using_plugin=True")
    plugin_args = traits.Dict({"qsub_args": "-q many"},
                                                      usedefault=True, desc='Plugin arguments.')
    test_mode = Bool(False, mandatory=False, usedefault=True,
                     desc='Affects whether where and if the workflow keeps its \
                            intermediary files. True to keep intermediary files. ')
    timeout = traits.Float(14.0)
    datagrabber = traits.Instance(Data, ())
    run_mode = traits.Enum("flame1","ols","flame12")
    save_script_only = traits.Bool(False)
    #Normalization
    brain_mask = traits.File(mandatory=True,desc='Brain Mask')
    name_of_project = traits.String("group_analysis",usedefault=True)
    do_randomize = traits.Bool(True)
    num_iterations = traits.Int(5000)

    #Correction:
    run_correction = traits.Bool(True)
    z_threshold = traits.Float(2.3)
    p_threshold = traits.Float(0.05)
    connectivity = traits.Int(26)

    # Advanced Options
    use_advanced_options = traits.Bool()
    advanced_script = traits.Code()

    # Buttons
    check_func_datagrabber = Button("Check")

def create_config():
    c = config()
    c.uuid = mwf.uuid
    c.datagrabber = Data(['copes','varcopes'])
    c.datagrabber.fields = []
    subs = DataBase()
    subs.name = 'subject_id'
    subs.values = ['sub01','sub02','sub03']
    subs.iterable = False
    fwhm = DataBase()
    fwhm.name='fwhm'
    fwhm.values=['0','6.0']
    fwhm.iterable = True
    con = DataBase()
    con.name='contrast'
    con.values=['con01','con02','con03']
    con.iterable=True
    c.datagrabber.fields.append(subs)
    c.datagrabber.fields.append(fwhm)
    c.datagrabber.fields.append(con)
    c.datagrabber.field_template = dict(copes='%s/preproc/output/fwhm_%s/cope*.nii.gz',
        varcopes='%s/preproc/output/fwhm_%s/varcope*.nii.gz')
    c.datagrabber.template_args = dict(copes=[['fwhm',"contrast","subject_id"]],
        varcopes=[['fwhm',"contrast","subject_id"]])
    return c

mwf.config_ui = create_config

"""
View
"""

def create_view():
    from traitsui.api import View, Item, Group
    from traitsui.menu import OKButton, CancelButton
    view = View(Group(Item(name='working_dir'),
                      Item(name='sink_dir'),
                      Item(name='crash_dir'),
                      label='Directories', show_border=True),
                Group(Item(name='run_using_plugin',enabled_when='save_script_only'),Item('save_script_only'),
                      Item(name='plugin', enabled_when="run_using_plugin"),
                      Item(name='plugin_args', enabled_when="run_using_plugin"),
                      Item(name='test_mode'), Item(name='timeout'),
                      label='Execution Options', show_border=True),
                Group(Item(name='datagrabber'),
                      label='Datagrabber', show_border=True),
                Group(Item(name='brain_mask'),Item("run_mode",enabled_when='not do_randomize'),
                      Item(name='do_randomize'),Item('num_iterations',enabled_when='do_randomize'),
                      label='Second Level', show_border=True),
                Group(Item("run_correction"),Item("z_threshold"),Item('p_threshold'),Item("connectivity"),
                    label='Correction', show_border=True),
                Group(Item(name='use_advanced_options'),
                    Item(name='advanced_script',enabled_when='use_advanced_options'),
                    label='Advanced',show_border=True),
                buttons=[OKButton, CancelButton],
                resizable=True,
                width=1050)
    return view

mwf.config_view = create_view

"""
Construct Workflow
"""

get_len = lambda x: len(x)

def create_2lvl(name="group"):
    import nipype.interfaces.fsl as fsl
    import nipype.pipeline.engine as pe
    import nipype.interfaces.utility as niu

    wk = pe.Workflow(name=name)
    
    inputspec = pe.Node(niu.IdentityInterface(fields=['copes','varcopes','brain_mask','run_mode']),name='inputspec')
    
    model = pe.Node(fsl.L2Model(),name='l2model')
    
    wk.connect(inputspec,('copes',get_len),model,'num_copes')
    
    mergecopes = pe.Node(fsl.Merge(dimension='t'),name='merge_copes')
    mergevarcopes = pe.Node(fsl.Merge(dimension='t'),name='merge_varcopes')
    
    flame = pe.Node(fsl.FLAMEO(),name='flameo')
    wk.connect(inputspec,"run_mode",flame,"run_mode")
    wk.connect(inputspec,'copes',mergecopes,'in_files')
    wk.connect(inputspec,'varcopes',mergevarcopes,'in_files')
    wk.connect(model,'design_mat',flame,'design_file')
    wk.connect(model,'design_con',flame, 't_con_file')
    wk.connect(mergecopes, 'merged_file', flame, 'cope_file')
    wk.connect(mergevarcopes,'merged_file',flame,'var_cope_file')
    wk.connect(model,'design_grp',flame,'cov_split_file')
    
    wk.connect(inputspec,'brain_mask',flame,'mask_file')

    outputspec = pe.Node(niu.IdentityInterface(fields=['zstat','tstat','cope',
                                                       'varcope','mrefvars',
                                                       'pes','res4d','mask',
                                                       'tdof','weights','pstat']),
                         name='outputspec')
                             
    wk.connect(flame,'copes',outputspec,'cope')
    wk.connect(flame,'var_copes',outputspec,'varcope')
    wk.connect(flame,'mrefvars',outputspec,'mrefvars')
    wk.connect(flame,'pes',outputspec,'pes')
    wk.connect(flame,'res4d',outputspec,'res4d')
    wk.connect(flame,'weights',outputspec,'weights')
    wk.connect(flame,'zstats',outputspec,'zstat')
    wk.connect(flame,'tstats',outputspec,'tstat')
    wk.connect(flame,'tdof',outputspec,'tdof')

    ztopval = pe.MapNode(interface=fsl.ImageMaths(op_string='-ztop',
        suffix='_pval'),
        name='z2pval',
        iterfield=['in_file'])

    wk.connect(flame,'zstats',ztopval,'in_file')
    wk.connect(ztopval,'out_file',outputspec,'pstat')

    return wk


def create_2lvl_rand(name="group_randomize",iters=5000):
    import nipype.interfaces.fsl as fsl
    import nipype.pipeline.engine as pe
    import nipype.interfaces.utility as niu

    wk = pe.Workflow(name=name)
    
    inputspec = pe.Node(niu.IdentityInterface(fields=['copes','varcopes','brain_mask']),name='inputspec')
    
    model = pe.Node(fsl.L2Model(),name='l2model')
    
    wk.connect(inputspec,('copes',get_len),model,'num_copes')
    
    mergecopes = pe.Node(fsl.Merge(dimension='t'),name='merge_copes')
    mergevarcopes = pe.Node(fsl.Merge(dimension='t'),name='merge_varcopes')
    
    rand = pe.Node(fsl.Randomise(base_name='OneSampleT', raw_stats_imgs=True, tfce=True, num_perm=iters),name='randomize')

    wk.connect(inputspec,'copes',mergecopes,'in_files')
    wk.connect(inputspec,'varcopes',mergevarcopes,'in_files')
    wk.connect(model,'design_mat',rand,'design_mat')
    wk.connect(model,'design_con',rand, 'tcon')
    wk.connect(mergecopes, 'merged_file', rand, 'in_file')
    #wk.connect(model,'design_grp',rand,'cov_split_file')
    
    wk.connect(inputspec,'brain_mask',rand,'mask')

    outputspec = pe.Node(niu.IdentityInterface(fields=['f_corrected_p_files',
                                                       'f_p_files',
                                                       'fstat_files',
                                                       't_corrected_p_files',
                                                       't_p_files', 
                                                       'tstat_file','mask']),
                         name='outputspec')
                             
    wk.connect(rand,'f_corrected_p_files',outputspec,'f_corrected_p_files')
    wk.connect(rand,'f_p_files',outputspec,'f_p_files')
    wk.connect(rand,'fstat_files',outputspec,'fstat_files')
    wk.connect(rand,'t_corrected_p_files',outputspec,'t_corrected_p_files')
    wk.connect(rand,'t_p_files',outputspec,'t_p_files')
    wk.connect(rand,'tstat_files',outputspec,'tstat_file')

    return wk



def get_datagrabber(c):

    import nipype.pipeline.engine as pe
    import nipype.interfaces.io as nio
    datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id',
                                                             'fwhm',"contrast"],
                                                   outfields=['copes','varcopes']),
                         name="datagrabber")
    datasource.inputs.base_directory = c.base_dir
    datasource.inputs.template = '*'
    datasource.inputs.field_template = dict(
                                copes=c.copes_template,
                                varcopes=c.varcopes_template)
    datasource.inputs.template_args = dict(copes=[['fwhm',"contrast","subject_id"]],
                                           varcopes=[['fwhm',"contrast","subject_id"]])
    return datasource

def get_substitutions(contrast):
    subs = [('_fwhm','fwhm'),
            ('_contrast_%s'%contrast,''),
            ('output','')]
    for i in range(0,20):
        subs.append(('_z2pval%d'%i,''))
        subs.append(('_cluster%d'%i, ''))
        subs.append(('_showslice%d'%i,''))
        subs.append(('_overlay%d/x_view.png'%i,'zstat%d_x_view.png'%(i+1)))
        subs.append(('_overlay%d/y_view.png'%i,'zstat%d_y_view.png'%(i+1)))
        subs.append(('_overlay%d/z_view.png'%i,'zstat%d_z_view.png'%(i+1)))
        subs.append(('_fdr%d'%i,''))

    return subs

def connect_to_config(c):
    import nipype.pipeline.engine as pe
    import nipype.interfaces.io as nio
    if not c.do_randomize:
        wk = create_2lvl()
        wk.inputs.inputspec.run_mode = c.run_mode
    else:
        wk  =create_2lvl_rand(iters=c.num_iterations)

    wk.base_dir = c.working_dir
    datagrabber = c.datagrabber.create_dataflow()  #get_datagrabber(c)
    #infosourcecon = pe.Node(niu.IdentityInterface(fields=["contrast"]),name="contrasts")
    #infosourcecon.iterables = ("contrast",c.contrasts)
    infosourcecon = datagrabber.get_node("contrast_iterable")
    #wk.connect(infosourcecon,'contrast',datagrabber,"contrast")
    sinkd = pe.Node(nio.DataSink(),name='sinker')
    sinkd.inputs.base_directory = c.sink_dir
    #sinkd.inputs.substitutions = [('_fwhm','fwhm'),('_contrast_','')]

    wk.connect(infosourcecon,("contrast",get_substitutions),sinkd,"substitutions")

    wk.connect(infosourcecon,"contrast",sinkd,"container")
    inputspec = wk.get_node('inputspec')
    outputspec = wk.get_node('outputspec')
    #datagrabber.inputs.subject_id = c.subjects
    #infosource = pe.Node(niu.IdentityInterface(fields=['fwhm']),name='fwhm_infosource')
    #infosource.iterables = ('fwhm',c.fwhm)
    #wk.connect(infosource,'fwhm',datagrabber,'fwhm')
    wk.connect(datagrabber,'datagrabber.copes', inputspec, 'copes')
    wk.connect(datagrabber,'datagrabber.varcopes', inputspec, 'varcopes')
    wk.inputs.inputspec.brain_mask = c.brain_mask
    if not c.do_randomize:
        wk.connect(outputspec,'cope',sinkd,'output.@cope')
        wk.connect(outputspec,'varcope',sinkd,'output.@varcope')
        wk.connect(outputspec,'mrefvars',sinkd,'output.@mrefvars')
        wk.connect(outputspec,'pes',sinkd,'output.@pes')
        wk.connect(outputspec,'res4d',sinkd,'output.@res4d')
        wk.connect(outputspec,'weights',sinkd,'output.@weights')
        wk.connect(outputspec,'zstat',sinkd,'output.@zstat')
        wk.connect(outputspec,'tstat',sinkd,'output.@tstat')
        wk.connect(outputspec,'pstat',sinkd,'output.@pstat')
        wk.connect(outputspec,'tdof',sinkd,'output.@tdof')

    if c.run_correction and not c.do_randomize:
        cluster = cluster_image()
        wk.connect(outputspec,"zstat",cluster,'inputspec.zstat')
        #wk.connect(outputspec,"mask",cluster,"inputspec.mask")
        #wk.connect(inputspec,"template",cluster,"inputspec.anatomical")
        cluster.inputs.inputspec.mask = c.brain_mask
        cluster.inputs.inputspec.zthreshold = c.z_threshold
        cluster.inputs.inputspec.pthreshold = c.p_threshold
        cluster.inputs.inputspec.connectivity = c.connectivity
        wk.connect(cluster,'outputspec.corrected_z',sinkd,'output.corrected.@zthresh')
        #wk.connect(cluster,'outputspec.slices',sinkd,'output.corrected.clusters')
        #wk.connect(cluster,'outputspec.cuts',sinkd,'output.corrected.slices')
        wk.connect(cluster,'outputspec.localmax_txt',sinkd,'output.corrected.@localmax_txt')
        wk.connect(cluster,'outputspec.index_file',sinkd,'output.corrected.@index')
        wk.connect(cluster,'outputspec.localmax_vol',sinkd,'output.corrected.@localmax_vol')

    if c.do_randomize:
        wk.connect(outputspec,'t_corrected_p_files',sinkd,'output.@t_corrected_p_files')
        wk.connect(outputspec,'t_p_files',sinkd,'output.@t_p_files')
        wk.connect(outputspec,'tstat_file',sinkd,'output.@tstat_file')


    return wk

mwf.workflow_function = connect_to_config

"""
Main
"""

def main(config_file):
    c = load_config(config_file, config)
    wk = connect_to_config(c)
    wk.config = {'execution': {'crashdump_dir': c.crash_dir, "job_finished_timeout":c.timeout}}
    
    if c.test_mode:
        wk.write_graph()
    if c.use_advanced_options:
        exec c.advanced_script

    from nipype.utils.filemanip import fname_presuffix
    wk.export(fname_presuffix(config_file,'','_script_').replace('.json',''))

    if c.run_using_plugin:
        wk.run(plugin=c.plugin,plugin_args=c.plugin_args)
    else:
        wk.run()
    return 1
    
mwf.workflow_main_function = main

"""
Register
"""

register_workflow(mwf)
