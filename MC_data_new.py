import ROOT as rt
import multiprocessing as mp
import os
import math
import optparse
import time
import sys
import numpy as np
import root_numpy as rn
from scipy.stats import expon
from scipy.stats import poisson
from collections import Counter
import random
from scipy.spatial.transform import Rotation as R

#sys.path.append("../reconstruction")
import swiftlib as sw


## FUNCTIONS DEFINITION

def NelGEM2(energyDep,z_hit,options):
    n_ioniz_el=energyDep/options.ion_pot
    drift_l = np.abs(z_hit-options.z_gem)
    n_ioniz_el_mean = np.abs(n_ioniz_el*np.exp(-drift_l/options.absorption_l)) 
    primary=poisson(n_ioniz_el_mean)           # poisson distribution for primary electrons
    n_ioniz_el=primary.rvs()                   # number of primary electrons
    n_el_oneGEM=0                              # number of secondary electrons
    gain1=expon(loc=0,scale=GEM1_gain)  # exponential distribution for the GAIN in the first GEM foil
    for k in range(0,n_ioniz_el):
        nsec = gain1.rvs()*extraction_eff_GEM1      # number of secondary electrons in the first GEM multiplication for each ionization electron
        n_el_oneGEM += nsec

    # total number of secondary electrons considering the gain in the 2nd GEM foil
    n_tot_el=n_el_oneGEM*GEM2_gain*extraction_eff_GEM2


    return n_tot_el

def cloud_smearing3D(x_hit,y_hit,z_hit,energyDep_hit,options):
    X=list(); Y=list(); Z=list() 
    X*=0; Y*=0; Z*=0
    nel=NelGEM2(energyDep_hit,z_hit,options)

    ## arrays of positions of produced electrons after GEM2
    X=(np.random.normal(loc=(x_hit), scale=np.sqrt(options.diff_const_sigma0T+options.diff_coeff_T*(np.abs(z_hit-options.z_gem))/10.), size=int(nel)))
    Y=(np.random.normal(loc=(y_hit), scale=np.sqrt(options.diff_const_sigma0T+options.diff_coeff_T*(np.abs(z_hit-options.z_gem))/10.), size=int(nel)))
    Z=(np.random.normal(loc=(z_hit-z_ini), scale=np.sqrt(options.diff_const_sigma0L+options.diff_coeff_L*(np.abs(z_hit-options.z_gem))/10.), size=int(nel)))   
    #print("distance from the GEM : %f cm"%((np.abs(z_hit-opt.z_gem))/10.))
    return X, Y, Z

def ph_smearing2D(x_hit,y_hit,z_hit,energyDep_hit,options):
    X=list(); Y=list()
    X*=0; Y*=0
    ## electrons in GEM2
    nel = NelGEM2(energyDep_hit,z_hit,options)
    ## photons in GEM3
    nph = nel * GEM3_gain *omega * options.photons_per_el * options.counts_per_photon
    ## arrays of positions of produced photons
    X=(np.random.normal(loc=(x_hit), scale=np.sqrt(options.diff_const_sigma0T+options.diff_coeff_T*(np.abs(z_hit-options.z_gem))/10.), size=int(nph)))
    Y=(np.random.normal(loc=(y_hit), scale=np.sqrt(options.diff_const_sigma0T+options.diff_coeff_T*(np.abs(z_hit-options.z_gem))/10.), size=int(nph)))
    return X, Y

    
def Nph_saturation(histo_cloud,options):
    Nph_array = np.zeros((histo_cloud.GetNbinsX(),histo_cloud.GetNbinsY()))
    Nph_tot = 0
    for i in range(options.x_vox_min,options.x_vox_max):
        for j in range(options.y_vox_min,options.y_vox_max):
            hout = 0
            for k in range(1,histo_cloud.GetNbinsZ()+1):
                hin = histo_cloud.GetBinContent(i,j,k)
                nel_in = hin
                hout += (nel_in * options.A * GEM3_gain)/(1 + options.beta * GEM3_gain  * nel_in) 
                
    
            nmean_ph= hout * omega * options.photons_per_el * options.counts_per_photon     # mean total number of photons
            photons=poisson(nmean_ph)                    # poisson distribution for photons
            n_ph=photons.rvs()  
            Nph_array[i-1][j-1] = n_ph
            Nph_tot += Nph_array[i-1][j-1]
            #if hout>0:
            #    print("number final electrons per voxel: %f"%hout)
            
    return Nph_tot, Nph_array



def AddBckg(options, i):
    bckg_array=np.zeros((options.x_pix,options.y_pix))
    if options.bckg:
        if sw.checkfiletmp(int(options.noiserun)):
            #options.tmpname = "/tmp/histograms_Run%05d.root" % int(options.noiserun)
            #options.tmpname = "/mnt/ssdcache/histograms_Run%05d.root" % int(options.noiserun)
            options.tmpname = "/nfs/cygno/users/dimperig/CYGNO/CYGNO-tmp/histograms_Run%05d.root" % int(options.noiserun)
            #FIXME
            #options.tmpname = "/nfs/cygno/users/dimperig/CYGNO/CYGNO-tmp/histograms_Run%05d_cropped.root" % int(options.noiserun)
        else:
            print ('Downloading file: ' + sw.swift_root_file(options.tag, int(options.noiserun)))
            options.tmpname = sw.swift_download_root_file(sw.swift_root_file(options.tag, int(options.noiserun)),int(options.noiserun))
        tmpfile =rt.TFile.Open(options.tmpname)
        tmphist = tmpfile.Get("pic_run%05d_ev%d"% (int(options.noiserun),i))
        bckg_array = rn.hist2array(tmphist) 
    #print(bckg_array)
    
    return bckg_array



def SaveValues(par, out):

    out.cd()
    out.mkdir('param_dir')
    #gDirectory.pwd()
    rt.gDirectory.ls()
    out.cd('param_dir')
    
    for k,v in par.items():
        if (k!='tag' and k!='energy_list' and k!='atom_list'):
            h=rt.TH1F(k, '', 1, 0, 1)
            h.SetBinContent(1, v)
            h.Write()
    out.cd()

    return None

def SaveEventInfo(info_dict, folder, out):
    
    out.cd()
    #gDirectory.pwd()
    out.cd('event_info')

    for k,v in info_dict.items():
        h=rt.TH1F(k, '', 1,0,1)
        h.SetBinContent(1,v)
        h.Write()
    out.cd()
    info_dict.clear()

    return None

######################################### MAIN EXECUTION ###########################################

if __name__ == "__main__":

    #CALL: python img_generator.py ConfigFile.txt -I <<INPUT_FOLDER>>

    parser = optparse.OptionParser("usage: %prog [options] arg1 arg2")

    parser.add_option("-I", "--inputfolder", dest="infolder", default=os.getcwd()+"/src/", help="specify the folder containing input files")
    parser.add_option("-O", "--outputfolder", dest="outfolder", default=os.getcwd()+"/out/", help="specify the output destination folder")
 
    (opt, args) = parser.parse_args()

    config = open(args[0], "r")         #GET CONFIG FILE
    params = eval(config.read())         #READ CONFIG FILE

    for k,v in params.items():
        setattr(opt, k, v)

    if opt.rootfiles==False :
        print('NR energies: '+str(opt.energy_list))
        print('Elements: '+str(opt.atom_list))

    ## fit from Fernando Amaro's single GEM gain measurement
    GEM1_gain = 0.0347*np.exp((0.0209)*opt.GEM1_HV)
    GEM2_gain = 0.0347*np.exp((0.0209)*opt.GEM2_HV)
    GEM3_gain = 0.0347*np.exp((0.0209)*opt.GEM3_HV)
    print("GEM1_gain = %d"%GEM1_gain)
    print("GEM2_gain = %d"%GEM2_gain)
    print("GEM3_gain = %d"%GEM3_gain)
    
    ## dividing Fernando's to Francesco&Karolina's single GEM gain measurement
    extraction_eff_GEM1 = 0.87319885*np.exp(-0.0020000000*opt.GEM1_HV)
    extraction_eff_GEM2 = 0.87319885*np.exp(-0.0020000000*opt.GEM2_HV)
    extraction_eff_GEM3 = 0.87319885*np.exp(-0.0020000000*opt.GEM3_HV)
    print("extraction eff GEM1 = %f" % extraction_eff_GEM1 )
    print("extraction eff GEM2 = %f" % extraction_eff_GEM2 )
    print("extraction eff GEM3 = %f" % extraction_eff_GEM3 )

    demag=opt.y_dim/opt.sensor_size
    a=opt.camera_aperture
    omega=1./math.pow((4*(demag+1)*a),2)   # solid angle ratio
    #print(omega)

#### CODE EXECUTION ####
    run_count=1
    t0=time.time()
    
    eventnumber = np.array([-999], dtype="int")
    particle_type = np.array([-999], dtype="int")
    energy_ini = np.array([-999], dtype="float32")
    ioniz_energy = np.array([-999], dtype="float32")
    drift = np.array([-999], dtype="float32")
    theta_ini = np.array([-999], dtype="float32")
    phi_ini = np.array([-999], dtype="float32")
    track_length_3D=np.empty((1),dtype="float32")
    x_vertex=np.empty((1),dtype="float32")
    y_vertex=np.empty((1),dtype="float32")
    z_vertex=np.empty((1),dtype="float32")
    x_vertex_end=np.empty((1),dtype="float32")
    y_vertex_end=np.empty((1),dtype="float32")
    z_vertex_end=np.empty((1),dtype="float32")
    proj_track_2D= np.empty((1),dtype="float32")

    z_ini = 0

    if not os.path.exists(opt.outfolder): #CREATING OUTPUT FOLDER
        os.makedirs(opt.outfolder)
            
    for infile in os.listdir(opt.infolder): #READING INPUT FOLDER
            
        if opt.rootfiles==True: #GEANT4
                # code to be used with input root files from Geant
            if infile.endswith('.root'):    #KEEPING .ROOT FILES ONLY
                
                #FIXME
                z_ini = 255.
                zbins = int(opt.zcloud/opt.z_vox_dim)
                rootfile=rt.TFile.Open(opt.infolder+infile)
                tree=rootfile.Get('nTuple')            #GETTING NTUPLES
            
                infilename=infile[:-5]    
                outfile=rt.TFile('%s/histograms_Run%05d.root' % (opt.outfolder,run_count), 'RECREATE') #OUTPUT NAME
                outfile.mkdir('event_info')
                SaveValues(params, outfile) ## SAVE PARAMETERS OF THE RUN
                outtree = rt.TTree("info_tree", "info_tree")
                outtree.Branch("eventnumber", eventnumber, "eventnumber/I")
                outtree.Branch("particle_type", particle_type, "particle_type/I")
                outtree.Branch("energy_ini", energy_ini, "energy_ini/F")
                outtree.Branch("theta_ini", theta_ini, "theta_ini/F")
                outtree.Branch("phi_ini", phi_ini, "phi_ini/F")

                final_imgs=list();
                
                if opt.events==-1:
                    totev=tree.GetEntries()
                else:
                    
                    if opt.events<=tree.GetEntries():
                        totev=opt.events
                    else:
                        totev=tree.GetEntries()
                    

    
                for entry in range(0, totev): #RUNNING ON ENTRIES
                    tree.GetEntry(entry)
                    eventnumber[0] = tree.eventnumber
                    #FIXME
                    particle_type[0] = 0
                    energy_ini[0] = tree.ekin_particle[0]*1000
                    phi_ini[0] = -999.
                    theta_ini[0] = -999.
                    phi_ini[0] = np.arctan2( (tree.y_hits[1]-tree.y_hits[0]),(tree.z_hits[1]-tree.z_hits[0]) )
                    theta_ini[0] = np.arccos( (tree.x_hits[1]-tree.x_hits[0]) / np.sqrt( np.power((tree.x_hits[1]-tree.x_hits[0]),2) + np.power((tree.y_hits[1]-tree.y_hits[0]),2) + np.power((tree.z_hits[1]-tree.z_hits[0]),2)) )
                    outtree.Fill()
                    final_image=rt.TH2I('pic_run'+str(run_count)+'_ev'+str(entry), '', opt.x_pix, 0, opt.x_pix-1, opt.y_pix, 0, opt.y_pix-1) #smeared track with background

                   
                    histname = "histo_cloud_pic_"+str(run_count)+"_ev"+str(int(entry)) 
                    histo_cloud = rt.TH3I(histname,"",opt.x_pix,0,opt.x_pix-1,opt.y_pix,0,opt.y_pix-1,zbins,0,zbins)
                    signal=rt.TH2I('sig_pic_run'+str(run_count)+'_ev'+str(entry), '', opt.x_pix, 0, opt.x_pix-1, opt.y_pix, 0, opt.y_pix-1) 

                    #print("created histo_cloud")
                    
                    ## with saturation
                    if (opt.saturation):
                        tot_el_G2 = 0
                        for ihit in range(0,tree.numhits):
                            #print("Processing hit %d of %d"%(ihit,tree.numhits))

                            ## here swapping X with Z beacuse in geant the drift axis is X
                            S3D = cloud_smearing3D(tree.z_hits[ihit],tree.y_hits[ihit],tree.x_hits[ihit],tree.energyDep_hits[ihit],opt)

                            for j in range(0, len(S3D[0])):
                                histo_cloud.Fill((0.5*opt.x_dim+S3D[0][j])*opt.x_pix/opt.x_dim, (0.5*opt.y_dim+S3D[1][j])*opt.y_pix/opt.y_dim, (0.5*histo_cloud.GetNbinsZ()*opt.z_vox_dim+S3D[2][j])/opt.z_vox_dim ) 
                                tot_el_G2+=1
                                
                        #tot_el_G2 = histo_cloud.Integral()
                    
                        # 2d map of photons applying saturation effect
                        result_GEM3 = Nph_saturation(histo_cloud,opt)
                        array2d_Nph = result_GEM3[1]
                        #tot_ph_G3 = result_GEM3[0] 
                        tot_ph_G3 = np.sum(array2d_Nph)

                        #print("tot num of sensor counts after GEM3 including saturation: %d"%(tot_ph_G3))
                        #print("tot num of sensor counts after GEM3 without saturation: %d"%(opt.A*tot_el_G2*GEM3_gain* omega * opt.photons_per_el * opt.counts_per_photon))
                        #print("Gain GEM3 = %f   Gain GEM3 saturated = %f"%(GEM3_gain, tot_ph_G3/(opt.A * tot_el_G2*omega * opt.photons_per_el * opt.counts_per_photon) ))   

                    ## no saturation
                    else:
                        tot_ph_G3=0
                        for ihit in range(0,tree.numhits):
                            ## here swapping X with Z beacuse in geant the drift axis is X
                            S2D = ph_smearing2D(tree.z_hits[ihit],tree.y_hits[ihit],tree.x_hits[ihit],tree.energyDep_hits[ihit],opt)
                            
                            for t in range(0, len(S2D[0])):
                                tot_ph_G3+=1

                                signal.Fill((0.5*opt.x_dim+S2D[0][t])*opt.x_pix/opt.x_dim, (0.5*opt.y_dim+S2D[1][t])*opt.y_pix/opt.y_dim ) 
                        array2d_Nph=rn.hist2array(signal)
                        array2d_Nph = array2d_Nph 
                        #print("tot num of sensor counts after GEM3 without saturation: %d"%(tot_ph_G3))


                    background=AddBckg(opt,entry)
                    total=array2d_Nph+background

                    final_image=rn.array2hist(total, final_image)
                    outfile.cd()
                    final_image.Write()            

                outfile.cd('event_info') 
                outtree.Write()
                print('COMPLETED RUN %d'%(run_count))
                run_count+=1
                #outfile.Close()

        if opt.rootfiles==False:    
            # code to be used with input txt files from SRIM
            #if infile.endswith('.txt'):    #KEEPING part.txt FILES ONLY NB: one single file with a specific name for the moment. there are other .txt files in the folder...this has to be fixed...

            infilename=infile[:-4]
            splitname=infilename.partition('ionization_profile_')[2]
            atom=splitname.partition('_')[0]
            inene=splitname.partition('_')[2].partition('keV')[0]

            if ((atom in opt.atom_list or len(opt.atom_list)==0) and (inene in opt.energy_list or len(opt.energy_list)==0)): #if list is empty, then do all
                #print('infile '+infile+', atom '+atom+', energy '+inene)

                #textfile=open(opt.infolder+infile, "r")
                textfile = np.loadtxt(opt.infolder+infile)[:,:]
                print("Opening file : %s" %infile )

                print('energy: '+inene+', atom: '+atom)
                if atom=='He': 
                    atom='2'
                    atom_mass='4'
                elif atom=='C': 
                    atom='6'
                    atom_mass='12'
                elif atom=='F': 
                    atom='9'
                    atom_mass='19'
                #outfile=rt.TFile(opt.outfolder+'/'+'histograms_Run'+atom+('%03d'%int(inene))+str(int(params['z_gem']/100.))+'.root', 'RECREATE') #OUTPUT NAME
                outfile=rt.TFile(opt.outfolder+'/'+'histograms_Run'+('%01d'%int(atom))+('%04d'%int(inene))+'.root', 'RECREATE') #OUTPUT NAME

#                outfile=rt.TFile('%s/histograms_Run%05d.root' % (opt.outfolder,run_count), 'RECREATE') #OUTPUT NAME
                outfile.mkdir('event_info')
                SaveValues(params, outfile) ## SAVE PARAMETERS OF THE RUN

                outtree = rt.TTree("info_tree", "info_tree")
                outtree.Branch("eventnumber", eventnumber, "eventnumber/I")
                outtree.Branch("particle_type", particle_type, "particle_type/I")
                outtree.Branch("energy_ini", energy_ini, "energy_ini/F")
                outtree.Branch("ioniz_energy", ioniz_energy, "ioniz_energy/F")
                outtree.Branch("drift",drift,"drift/F")
                outtree.Branch("theta_ini", theta_ini, "theta_ini/F")
                outtree.Branch("phi_ini", phi_ini, "phi_ini/F")
                outtree.Branch('x_vertex',x_vertex,"x_vertex/F")
                outtree.Branch('y_vertex',y_vertex,"y_vertex/F")
                outtree.Branch('z_vertex',z_vertex,"z_vertex/F")
                outtree.Branch('x_vertex_end',x_vertex_end,"x_vertex_end/F")
                outtree.Branch('y_vertex_end',y_vertex_end,"y_vertex_end/F")
                outtree.Branch('z_vertex_end',z_vertex_end,"z_vertex_end/F")
                outtree.Branch('proj_track_2D',proj_track_2D,"proj_track_2D/F")
                outtree.Branch('track_length_3D',track_length_3D,"track_length_3D/F")

                final_imgs=list();
                zvec=list(); yvec=list(); xvec=list(); evec=list(); ioniz_evec=list(); phiini=list(); thetaini=list(); eini=list();
                zvec*=0; yvec*=0; xvec*=0; evec*=0; ioniz_evec*=0; phiini*=0; thetaini*=0; eini*=0;
                #lastentry=0
                nhits=0
                iline=0
                #sumene = 0
                #sumeneQF = 0
                #content = textfile.readlines() #READ IN TXT FILE
                tot_el_G2=0
                tot_ph_G3=0
                zbins = int(opt.zcloud/opt.z_vox_dim)
                
                array2d_Nph = np.zeros((opt.x_pix,opt.y_pix))
                
                if opt.events==-1:
                    totev=len(Counter(textfile[:,0]).keys())
                else:
                    if opt.events<=len(Counter(textfile[:,0]).keys()):
                        totev=opt.events
                    else:
                        totev=len(Counter(textfile[:,0]).keys())

                for entry in range(0,totev): ###LOOP OVER IONS

                    M = R.random().as_matrix()
                    data = textfile[textfile[:,0]==entry+1,:]
                    rand_num = random.randrange(10,500)
                    xvec = M[0][0]*data[:,3]+M[0][1]*data[:,4]+M[0][2]*data[:,5]
                    yvec = M[1][0]*data[:,3]+M[1][1]*data[:,4]+M[1][2]*data[:,5]
                    zvec = M[2][0]*data[:,3]+M[2][1]*data[:,4]+M[2][2]*data[:,5]+rand_num
                    evec = data[:,6]
                    ioniz_evec = data[:,7]
                    prim_second = data[:,2]
                    nhits = len(data)


                    sumene = 0.
                    sumeneQF = 0.
                    proj_length_2D=0
                    proj_length_2D_new=0
                    track_len_3D=0
                    track_len_3D_new=0
                    track_3D_MC=0
                    proj_pos_2D=0
                    proj_pos_3D=0
                    
                    for i in range(0, nhits-1):
                        sumene += evec[i]
                        sumeneQF += ioniz_evec[i]
                        #pr_or_sec[i] = prim_second[i]
                        #ioniz_energy_hits[i] = ioniz_evec[i]

                        if data[i,2]==1 and data[i+1,2]==1:
                            proj_pos_2D = np.sqrt(pow((yvec[i+1]-yvec[i]),2)+ pow((xvec[i+1]-xvec[i]),2))
                            proj_pos_3D = np.sqrt(pow((yvec[i+1]-yvec[i]),2)+ pow((zvec[i+1]-zvec[i]),2)+pow((xvec[i+1]-xvec[i]),2))
                        #if i == 0:
                            #phi_ini[0] = np.arctan2( (yvec[i+1]-yvec[i]), (xvec[i+1]-xvec[i]) )
                            #theta_ini[0] = np.arccos( (zvec[i+1]-zvec[i]) / np.sqrt(pow((yvec[i+1]-yvec[i]),2)+ pow((zvec[i+1]-zvec[i]),2)+pow((xvec[i+1]-xvec[i]),2)))
                            
                        proj_length_2D += proj_pos_2D
                        track_len_3D += proj_pos_3D

                    eventnumber[0] = entry
                    particle_type[0]=int('100'+('%03d'%int(atom))+('%03d'%int(atom_mass))+'0')
                    energy_ini[0] = np.sum(evec)
                    ioniz_energy[0] = np.sum(ioniz_evec)
                    drift[0] = rand_num
                    phi_ini[0] = np.arctan2( yvec[0],xvec[0] ) 
                    theta_ini[0] = np.arccos( (zvec[0]-rand_num) / np.sqrt( np.power(xvec[0],2) + np.power(yvec[0],2) + np.power(zvec[0]-rand_num,2)) )
                    x_vertex[0]= (xvec[0]+0.5*opt.x_dim)*opt.x_pix/opt.x_dim
                    y_vertex[0]= (yvec[0]+0.5*opt.y_dim)*opt.y_pix/opt.y_dim
                    z_vertex[0]= zvec[0]
                    x_vertex_end[0]= (xvec[-1]+0.5*opt.x_dim)*opt.x_pix/opt.x_dim
                    y_vertex_end[0]= (yvec[-1]+0.5*opt.y_dim)*opt.y_pix/opt.y_dim
                    z_vertex_end[0]= zvec[-1]
                    track_length_3D[0]= track_len_3D
                    proj_track_2D[0]=proj_length_2D

                    outtree.Fill()

                    
                    # SATURATION SIM
                    if(opt.saturation):

                        histname = "histo_cloud"+str(run_count)+"_pic_"+str(int(entry))
                        histo_cloud = rt.TH3I(histname,"",opt.x_pix,0,opt.x_pix-1,opt.y_pix,0,opt.y_pix-1,zbins,0,zbins)
                        final_image=rt.TH2I('pic_run'+str(run_count)+'_ev0', '', opt.x_pix, 0, opt.x_pix-1, opt.y_pix, 0, opt.y_pix-1) #smeared track with background

                        for i in range(0, nhits-1):
                            S3D = cloud_smearing3D(xvec[i],yvec[i],zvec[i],ioniz_evec[i],opt)

                            for j in range(0, len(S3D[0])): #LOOP OVER ELECTRONS AFTER GEM2 

                                histo_cloud.Fill((0.5*opt.x_dim+S3D[0][j])*opt.x_pix/opt.x_dim, (0.5*opt.y_dim+S3D[1][j])*opt.y_pix/opt.y_dim, (0.5*opt.zcloud+S3D[2][j])*zbins/opt.zcloud )
                                tot_el_G2+=1

                                    #print("tot_el_G2 = %d"%tot_el_G2)
                
                        # 2d map of photons applying saturation effect
                        #thits=time.time()
                        #print('Loop over hits of event {0} took {1} seconds'.format(str(entry),str(thits-t0)))
                        result_GEM3 = Nph_saturation(histo_cloud,opt)
                        array2d_Nph = result_GEM3[1]
                        tot_ph_G3 = result_GEM3[0] #np.sum(array2d_Nph)
                                #print("tot num of sensor counts after GEM3 including saturation: %d"%(tot_ph_G3))
                                #print("tot num of sensor counts after GEM3 without saturation: %d"%(opt.A*tot_el_G2*GEM3_gain* omega * opt.photons_per_el * opt.counts_per_photon))
                                #print("Gain GEM3 = %f   Gain GEM3 saturated = %f"%(GEM3_gain, tot_ph_G3/(tot_el_G2*omega * opt.photons_per_el * opt.counts_per_photon) ))
                
                        # add background
                        background=AddBckg(opt,entry)
                        total=array2d_Nph+background
                        rn.array2hist(total, final_image)
                        # write output file
                        outfile.cd()
                        final_image.Write()
                        nphot = final_image.Integral()                                      
                
                                    #print("lastentry = %d"%lastentry)
                                    #print("tot ion energy = "+str(sumene))
                                    #print("ion energy * QF = "+str(sumeneQF))
                                    #print("nphot = "+str(nphot))
                                 
                        sumene = 0.
                        sumeneQF = 0.
                        zvec*=0; yvec*=0; xvec*=0; evec*=0; phiini*=0; thetaini*=0; eini*=0; #RESET LISTS
                        tot_el_G2 = 0  

                        ## CLOSE LOOP OVER GEM2 ELECTRONS
                    
                    # NO SATURATION 
                    else:

                        signal=rt.TH2I('sig_run'+str(run_count)+'_ev'+str(entry), '', opt.x_pix, 0, opt.x_pix-1, opt.y_pix, 0, opt.y_pix-1) #smeared track with background
                        final_image=rt.TH2I('pic_run'+str(run_count)+'_ev'+str(entry), '', opt.x_pix, 0, opt.x_pix-1, opt.y_pix, 0, opt.y_pix-1) #smeared track with background

                        for i in range(0, nhits-1):
                            S2D = ph_smearing2D(xvec[i],yvec[i],zvec[i],ioniz_evec[i],opt)
                            
                            for t in range(0, len(S2D[0])):
                                tot_ph_G3+=1

                                signal.Fill((0.5*opt.x_dim+S2D[0][t])*opt.x_pix/opt.x_dim, (0.5*opt.y_dim+S2D[1][t])*opt.y_pix/opt.y_dim ) 

                        # 2d map of photons applying saturation effect
                        array2d_Nph=rn.hist2array(signal)
                        #print("tot num of sensor counts after GEM3 without saturation: %d"%(tot_ph_G3))
                
                        background=AddBckg(opt,entry)
                        total=array2d_Nph+background
                                
                        rn.array2hist(total,final_image)
                        nphot = final_image.Integral()

                                # write output file
                        outfile.cd()
                        final_image.Write()
                        nphot = final_image.Integral()

                                #    print("lastentry = %d"%lastentry)
                                #    print("tot ion energy = "+str(sumene))
                                #    print("ion energy * QF = "+str(sumeneQF))
                                #    print("nphot = "+str(nphot))
                             
                        sumene = 0.
                        sumeneQF = 0.   
                        zvec*=0; yvec*=0; xvec*=0; evec*=0; phiini*=0; thetaini*=0; eini*=0; #RESET LISTS
                        tot_ph_G3 = 0    

                        ## CLOSE LOOP OVER GEM3 PHOTONS       
                             
                ## CLOSE LOOP OVER IONS 

            ## CONTINUE LOOP OVER TXT FILES
                print('COMPLETED RUN %d'%(run_count))
                run_count+=1
                outfile.cd('event_info') 
                outtree.Write()
                #outfile.Close()
           # else:
            #    print('File not found in input folder - maybe you need to change something?')
            #    print('Looking for {0}, {1} keV in {2}'.format(str(opt.atom_list),str(opt.energy_list),str(opt.infolder))) 

    t1=time.time()
    if opt.donotremove == False:
        sw.swift_rm_root_file(opt.tmpname)
    print('\n')
    print('Generation took %d seconds'%(t1-t0))

