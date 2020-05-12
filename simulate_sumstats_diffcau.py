import random, math, sys, argparse, time, logging, gzip
import scipy
import scipy.stats as ss
import pandas as pd
import numpy as np
import numpy.linalg as nplinalg
import pysnptools
from pysnptools.snpreader import Bed
from pysnptools.standardizer import Unit
from annot import *

# main function
def main():

    # get command line arguments
    args = get_command_line()

    cont_annot = args.cont_annot

    # start log
    init_log(args.out)

    # load annotation matrix
    annot_prefix = args.annot
    start_chrom = args.use_chrom[0]
    end_chrom = args.use_chrom[1]
    annot_snps, annot_names, annot_mat = load_annot(annot_prefix,
        start_chrom, end_chrom)
    annot_mat_std = annot_mat.copy()

    # standardize annotation
    if args.std_annot == True:
        annot_mat_std[:,1:] = (annot_mat_std[:,1:]-annot_mat_std[:,1:].mean(axis=0))/\
                              (annot_mat_std[:,1:].std(axis=0)+1e-16)

    # load coefficient to be simulated
    sim_coef = pd.read_table(args.sim_coef_file, delim_whitespace=True)
    sim_coef['THETA'] *= args.scale
    coef_mat = sim_coef.loc[:, ['TAU1', 'TAU2', 'THETA']].values

    # get per snp heritability and covariance
    persnp_hsq1 = np.dot(annot_mat_std, coef_mat[:,0])
    persnp_hsq2 = np.dot(annot_mat_std, coef_mat[:,1])
    persnp_cov = np.dot(annot_mat_std, coef_mat[:,2])
    persnp_hsq1[persnp_hsq1<0] = 0.0
    persnp_hsq2[persnp_hsq2<0] = 0.0
    persnp_cov[persnp_cov<0] = 0.0

    # iterate through the number of simulations
    for i in xrange(args.num_sim):
  
        out_info = pd.DataFrame()

        Z1, Z2 = sim_sumstats(args.bfile, annot_names, annot_mat, coef_mat,
                    cont_annot, args.pct_causal_shared, args.pct_causal_spec,
                    args.sim_hsq, args.nsample, persnp_hsq1, persnp_hsq2,
                    persnp_cov, out_info, i+1, start_chrom, end_chrom,
                    args.standardize, args.nbin, args.out, args.info_only,
                    args.scale)
        
        Z1.to_csv('{}{}_pop1.sumstats.gz'.format(args.out,i+1),
            compression='gzip', index=False, sep='\t')

        Z2.to_csv('{}{}_pop2.sumstats.gz'.format(args.out,i+1),
            compression='gzip', index=False, sep='\t')

    # end lot
    end_log()

# draw causal snps
def draw_causals(n01, n10, n11, nsnps):
    
    all_index = np.array(range(nsnps))
    cau01_idx = np.random.choice(range(nsnps),size=n01, replace=False)
    cau01 = all_index[cau01_idx]

    all_index = np.delete(all_index, cau01_idx)
    cau10_idx = np.random.choice(range(all_index.shape[0]),
        size=n10, replace=False)
    cau10 = all_index[cau10_idx]

    all_index = np.delete(all_index, cau10_idx)
    cau11_idx = np.random.choice(range(all_index.shape[0]),
        size=n11, replace=False)
    cau11 = all_index[cau11_idx]

    cau1 = np.sort(np.concatenate((cau01, cau11)))
    cau2 = np.sort(np.concatenate((cau10, cau11)))

    return (cau1, cau2)

def sim_sumstats(bfile, annot_names, annot_mat, coef_mat, cont_annot,
    pct_causal_shared, pct_causal_spec, hsq, nsample, persnp_hsq1,
    persnp_hsq2, persnp_cov, out_info, idx, start_chrom, end_chrom,
    standardize, nbin, out_nm, info_only, scale):

    # load legend
    bfile1, bfile2 = bfile
    legend = [] 
    for i in xrange(start_chrom, end_chrom+1):
        legend_chrom = pd.read_table('{}{}.bim'.format(bfile1,i), header=None,
            delim_whitespace=True, usecols=[0,1,3,4,5])
        legend_chrom.columns = ['CHR', 'SNP', 'BP', 'A2', 'A1']
        legend_chrom = legend_chrom[['CHR', 'SNP', 'BP', 'A1', 'A2']]
        legend.append(legend_chrom)
    legend = pd.concat(legend, axis=0, ignore_index=True)

    # load frequency file
    freq1 = []
    freq2 = []
    for i in xrange(start_chrom, end_chrom+1):
        freq1_chrom = pd.read_table('{}{}.frq'.format(bfile1,i),
            delim_whitespace=True)
        freq2_chrom = pd.read_table('{}{}.frq'.format(bfile2,i),
            delim_whitespace=True)
        freq1.append(freq1_chrom)
        freq2.append(freq2_chrom)
    freq1 = pd.concat(freq1, axis=0, ignore_index=True)
    freq2 = pd.concat(freq2, axis=0, ignore_index=True)
    maf5_idx= np.where((freq1['MAF']>0.05) & (freq2['MAF']>0.05))[0]

    # draw index of causal snps
    maf_thres = 0.001
    all_idx = np.arange(legend.shape[0])
    nsnps_tot = legend.shape[0] 
    ncau_shared = int(np.floor(np.float(nsnps_tot)*pct_causal_shared))
    ncau_spec = int(np.floor(np.float(nsnps_tot)*pct_causal_spec))
    cau_idx1, cau_idx2 = draw_causals(ncau_spec, ncau_spec,
        ncau_shared, nsnps_tot)

    # draw effect size
    persnp_hsq1[freq1['MAF']<=maf_thres] = 0.0
    persnp_hsq2[freq2['MAF']<=maf_thres] = 0.0
    persnp_cov[(freq1['MAF']<=maf_thres) |
               (freq2['MAF']<=maf_thres)] = 0.0
    beta1, beta2 = sim_beta(persnp_hsq1, persnp_hsq2, persnp_cov, scale)

    # set non causal to zero
    mask1 = np.zeros(beta1.shape[0], dtype=bool)
    mask1[cau_idx1] = True
    beta1[~mask1] = np.float32(0.0)
   
    mask2 = np.zeros(beta2.shape[0], dtype=bool)
    mask2[cau_idx2] = True
    beta2[~mask2] = np.float32(0.0)

    # scale effect size to match heritability
    var_beta1 = np.sum(np.square(beta1[maf5_idx]))
    var_beta2 = np.sum(np.square(beta2[maf5_idx]))
    cov_beta12 = np.sum(beta1[maf5_idx]*beta2[maf5_idx])
    
    hsq1, hsq2 = hsq
    
    ratio = hsq1 / var_beta1
    beta1 = beta1 * np.sqrt(ratio)
    ratio = hsq2 / var_beta2
    beta2 = beta2 * np.sqrt(ratio)
    var_beta1 = np.sum(np.square(beta1))
    var_beta2 = np.sum(np.square(beta2))
    cov_beta12 = np.sum(beta1*beta2)
    cor_beta12 = cov_beta12 / np.sqrt(var_beta1*var_beta2)

    # record simulated info for binary annotations
    nannot = annot_mat.shape[1]
    beta1_sq = np.square(beta1)
    beta2_sq = np.square(beta2)
    beta12_prod = beta1 * beta2
    beta1_sq_maf5 = beta1_sq[maf5_idx]
    beta2_sq_maf5 = beta2_sq[maf5_idx]
    beta12_prod_maf5 = beta12_prod[maf5_idx]
    annot_mat_maf5 = annot_mat[maf5_idx,:]
    nsnp_tot = annot_mat_maf5.shape[0]
    for i in xrange(nannot):
        nm = annot_names.loc[i, 'ANNOT']
        nsnp_annot = np.sum(annot_mat_maf5[:,i])
        hsq1_annot = np.dot(annot_mat_maf5[:,i].T, beta1_sq_maf5)
        hsq2_annot = np.dot(annot_mat_maf5[:,i].T, beta2_sq_maf5)
        cov_annot =  np.dot(annot_mat_maf5[:,i].T, beta12_prod_maf5)
        cor_annot = cov_annot / np.sqrt(hsq1_annot*hsq2_annot)
        out_info.loc[i,'ANNOT'] = nm
        out_info.loc[i,'NSNP'] = nsnp_annot
        out_info.loc[i,'HSQ1'] = hsq1_annot
        out_info.loc[i,'HSQ1_ENRICHMENT'] = hsq1_annot/out_info.loc[0,'HSQ1']\
            *nsnp_tot/nsnp_annot
        out_info.loc[i,'HSQ2'] = hsq2_annot
        out_info.loc[i,'HSQ2_ENRICHMENT'] = hsq2_annot/out_info.loc[0,'HSQ2']\
            *nsnp_tot/nsnp_annot
        out_info.loc[i,'GCOV'] = cov_annot
        out_info.loc[i,'GCOV_ENRICHMENT'] = cov_annot/out_info.loc[0,'GCOV']\
            *nsnp_tot/nsnp_annot
        out_info.loc[i,'GCOR'] = cor_annot
        out_info.loc[i,'GCORSQ'] = cor_annot*cor_annot
        out_info.loc[i,'GCORSQ_ENRICHMENT'] = cor_annot*cor_annot \
            /out_info.loc[0,'GCORSQ']

    # for continuous annotations
    if cont_annot != None:

        # create binary annotations based on continuous annotations values
        bin_annot_names, bin_annot_mat = get_bin_annot_mat(annot_mat_maf5,
            annot_names, cont_annot, nbin)

        # record simulated info for continuous annotations
        bin_nannot = len(bin_annot_names)
        for i in xrange(bin_nannot):
            nm = bin_annot_names[i]
            nsnp_annot = np.sum(bin_annot_mat[:,i])
            hsq1_annot = np.dot(bin_annot_mat[:,i].T, beta1_sq_maf5)
            hsq2_annot = np.dot(bin_annot_mat[:,i].T, beta2_sq_maf5)
            cov_annot = np.dot(bin_annot_mat[:,i].T, beta12_prod_maf5)
            cor_annot = cov_annot / np.sqrt(hsq1_annot*hsq2_annot)
            out_info.loc[i+nannot,'ANNOT'] = nm
            out_info.loc[i+nannot,'NSNP'] = nsnp_annot
            out_info.loc[i+nannot,'HSQ1'] = hsq1_annot
            out_info.loc[i+nannot,'HSQ1_ENRICHMENT'] = \
                hsq1_annot/out_info.loc[0,'HSQ1']*nsnp_tot/nsnp_annot
            out_info.loc[i+nannot,'HSQ2'] = hsq2_annot
            out_info.loc[i+nannot,'HSQ2_ENRICHMENT'] = \
                hsq2_annot/out_info.loc[0,'HSQ2']*nsnp_tot/nsnp_annot
            out_info.loc[i+nannot,'GCOV'] = cov_annot
            out_info.loc[i+nannot,'GCOV_ENRICHMENT'] = \
                cov_annot/out_info.loc[0,'GCOV']*nsnp_tot/nsnp_annot
            out_info.loc[i+nannot,'GCOR'] = cor_annot
            out_info.loc[i+nannot,'GCORSQ'] = cor_annot*cor_annot
            out_info.loc[i+nannot,'GCORSQ_ENRICHMENT'] = cor_annot*cor_annot \
                /out_info.loc[0,'GCORSQ']

    # write out info to disk
    out_info.to_csv('{}{}.info'.format(out_nm,idx),sep='\t',index=False)
    if info_only == True:
        sys.exit()

    nsample1, nsample2 = nsample

    # simulate phenotype and summary stats for population 1
    beta1 = beta1.astype(np.float32)
    pheno1 = sim_pheno(bfile1, nsample1, start_chrom, end_chrom, cau_idx1,
        beta1, legend, standardize)
    zsc_maf_thres = 0.01

    logging.info('Phenotype for population 1 simulated')
   
    out1_idx = np.where(freq1['MAF']>zsc_maf_thres)[0]
    
    Z1 = sim_zsc(bfile1, nsample1, start_chrom, end_chrom, pheno1,
        legend, standardize, freq1['MAF'].values)
    Z1_df = legend.loc[out1_idx,:].copy().reset_index(drop=True)
    Z1_df['Z'] = Z1; Z1_df['N'] = pheno1.shape[0]
    Z1_df = Z1_df[['SNP', 'CHR', 'BP', 'A1', 'A2', 'Z', 'N']]
    Z1_df['CHR'] = Z1_df['CHR'].astype(np.int32)

    logging.info('Summary stats for population 1 simulated')

    # simulate phenotype and summary stats for population 2
    beta2 = beta2.astype(np.float32)
    pheno2 = sim_pheno(bfile2, nsample2, start_chrom, end_chrom, cau_idx2,
        beta2, legend, standardize)
    
    logging.info('Phenotype for population 2 simulated')
   
    out2_idx = np.where(freq2['MAF']>zsc_maf_thres)[0]
    Z2 = sim_zsc(bfile2, nsample2, start_chrom, end_chrom, pheno2,
        legend, standardize, freq2['MAF'].values)
    Z2_df = legend.loc[out2_idx,:].copy().reset_index(drop=True)
    Z2_df['Z'] = Z2; Z2_df['N'] = pheno2.shape[0]
    Z2_df = Z2_df[['SNP', 'CHR', 'BP', 'A1', 'A2', 'Z', 'N']]
    Z2_df['CHR'] = Z2_df['CHR'].astype(np.int32)

    logging.info('Summary stats for population 2 simulated')

    return Z1_df, Z2_df

def create_block(start_idx, stop_idx, nblock):

    block_size = int(np.ceil(float(stop_idx-start_idx+1)/float(nblock)))
    cuts = range(start_idx, stop_idx, block_size)
    blocks = []
    
    for i in range(len(cuts)-1):
        start = cuts[i]
        stop = cuts[i+1]
        blocks.append(np.arange(start, stop))
        
    blocks.append(np.arange(cuts[len(cuts)-1], stop_idx+1))

    return blocks

def sim_pheno(bfile, nsample, start_chrom, end_chrom, cau_idx, beta, legend,
    standardize, nblock=40):
    
    mask = np.zeros(beta.shape[0], dtype=bool)
    mask[cau_idx] = True

    fam = '{}{}.fam'.format(bfile, start_chrom)
    nindv = pd.read_table(fam, nrows=nsample, header=None).shape[0]
    pheno = np.zeros(nindv, dtype=np.float32)
    
    for i in xrange(start_chrom, end_chrom+1):
        
        snpdata = Bed('{}{}.bed'.format(bfile, i), count_A1=False)
        nsnp = snpdata.sid_count
        blocks = create_block(0, nsnp-1, nblock)

        snp_idx = np.where(legend['CHR']==i)[0]
        beta_chrom = beta[snp_idx]
        mask_chrom = mask[snp_idx]
        
        for blk in blocks:
            mask_chrom_blk = mask_chrom[blk]
            use_idx = blk[mask_chrom_blk==True]
            
            snpdata_blk = snpdata[0:nindv,use_idx]
            if standardize == False:
                snpdata_blk = snpdata_blk.read(dtype=np.float32).val
            else:
                snpdata_blk = snpdata_blk.read(dtype=np.float32)\
                    .standardize(Unit()).val
            if standardize == False:
                snpdata_blk -= snpdata_blk.mean(axis=0)
            
            pheno += np.dot(snpdata_blk, beta_chrom[use_idx])
    
    sigma_e = np.sqrt(1.0-np.var(pheno))
    
    eps = np.random.normal(scale=sigma_e, size=nindv).astype(np.float32)
    pheno += eps

    return pheno

def sim_zsc(bfile, nsample, start_chrom, end_chrom, pheno, legend,
    standardize, freq, nblock=40):
    
    zsc_maf_thres = 0.01

    nindv = nsample

    nsnp_all = legend.shape[0]
    zsc = np.zeros(nsnp_all, dtype=np.float32)

    for i in xrange(start_chrom, end_chrom+1):
        
        snpdata = Bed('{}{}.bed'.format(bfile, i), count_A1=False)
        nsnp = snpdata.sid_count
        blocks = create_block(0, nsnp-1, nblock)

        snp_idx = np.where(legend['CHR']==i)[0]
        zsc_chrom = np.zeros(snp_idx.shape[0])
        
        freq_chrom = freq[snp_idx]
        mask_chrom = np.zeros(nsnp, dtype=bool)
        mask_chrom[freq_chrom>zsc_maf_thres] = True

        for blk in blocks:
            
            mask_chrom_blk = mask_chrom[blk]
            use_idx = blk[mask_chrom_blk==True]

            snpdata_blk = snpdata[0:nindv,use_idx]
            if standardize == False:
                snpdata_blk = snpdata_blk.read(dtype=np.float32).val
            else:
                snpdata_blk = snpdata_blk.read(dtype=np.float32)\
                    .standardize(Unit()).val
            if standardize == False:
                snpdata_blk -= snpdata_blk.mean(axis=0)
            if standardize == True:
                zsc_chrom[use_idx] = np.dot(snpdata_blk.T, pheno)/np.sqrt(nindv)
            else:
                sigmasq = snpdata_blk.var(axis=0)
                zsc_chrom[use_idx] = np.dot(snpdata_blk.T,pheno)
                zsc_chrom[use_idx] /= np.sqrt(nindv*sigmasq)
        
        zsc[snp_idx] = zsc_chrom

    return zsc[freq>zsc_maf_thres]

def sim_beta(persnp_hsq1, persnp_hsq2, persnp_cov, scale):
    
    # initialize beta
    nsnp = persnp_hsq1.shape[0]
    beta1 = np.zeros(nsnp)
    beta2 = np.zeros(nsnp)
    
    # find snps with non zero variance
    non_zero_idx12 = np.where((persnp_hsq1>0) &
                              (persnp_hsq2>0) &
                              (persnp_cov>0))[0]
    persnp_hsq1_nz12 = persnp_hsq1[non_zero_idx12]
    persnp_hsq2_nz12 = persnp_hsq2[non_zero_idx12]
    persnp_cov_nz12 = persnp_cov[non_zero_idx12]
    S = persnp_hsq2_nz12 - np.square(persnp_cov_nz12)/persnp_hsq1_nz12
    persnp_cov_nz12[S<0] = np.sign(persnp_cov_nz12[S<0]) * \
        persnp_hsq1_nz12[S<0] * persnp_hsq2_nz12[S<0]

    # find cholesky
    S = persnp_hsq2_nz12 - np.square(persnp_cov_nz12)/persnp_hsq1_nz12
    S[S<0] = 0.0
    L11 = np.sqrt(persnp_hsq1_nz12)
    L12 = persnp_cov_nz12 / np.sqrt(persnp_hsq1_nz12)
    L22 = np.sqrt(S)

    # draw from normal
    nsnp_sim = non_zero_idx12.shape[0]
    tmp1 = np.random.normal(size=nsnp_sim)
    tmp2 = np.random.normal(size=nsnp_sim)

    # multiply cholesky
    beta1_nz12 = L11 * tmp1
    beta2_nz12 = L12 * tmp1 + L22 * tmp2
 
    # when gcor is one
    if scale == 1:
        beta2_nz12 = beta1_nz12.copy()

    # update beta
    beta1[non_zero_idx12] = beta1_nz12
    beta2[non_zero_idx12] = beta2_nz12

    # simulate rest of snps
    non_zero_idx1 = np.where((persnp_hsq1>0) &
                             ((persnp_hsq2<=0) |
                              (persnp_cov<=0)))[0]
    non_zero_idx2 = np.where((persnp_hsq2>0) &
                             ((persnp_hsq1<=0) |
                              (persnp_cov<=0)))[0]
    nsnp_sim1 = non_zero_idx1.shape[0]
    nsnp_sim2 = non_zero_idx2.shape[0]
    persnp_hsq1_nz1 = persnp_hsq1[non_zero_idx1]
    persnp_hsq2_nz2 = persnp_hsq2[non_zero_idx2]
    beta1_nz1 = np.sqrt(persnp_hsq1_nz1)*np.random.normal(size=nsnp_sim1)
    beta2_nz2 = np.sqrt(persnp_hsq2_nz2)*np.random.normal(size=nsnp_sim2)

    # update beta
    beta1[non_zero_idx1] = beta1_nz1
    beta2[non_zero_idx2] = beta2_nz2

    return beta1, beta2

# create binary annotations based on continuous values
def get_bin_annot_mat(annot_mat, all_annot, names, nbins):
    
    # initialize the new names
    new_names = []
    for j in xrange(len(names)):
        nm = names[j]
        for i in xrange(nbins):
            new_names.append('{}{}'.format(nm,i+1))

    # initialize new annot mat
    nsnp = annot_mat.shape[0]
    new_annot_mat = np.zeros((nsnp, len(new_names)), dtype=np.float32)

    # create the new annot mat
    idx = 0
    for j in xrange(len(names)):
        nm = names[j]
        nm_idx = np.where(all_annot['ANNOT']==nm)[0][0]
        annot_vec = annot_mat[:,nm_idx]
        cut = pd.qcut(annot_vec, nbins, retbins=True)[1]
        for i in range(cut.shape[0]-1):
            low = cut[i]
            if i == 0: low -= 0.00001 
            high = cut[i+1]
            updt_idx = np.where((annot_vec>low) &
                                (annot_vec<=high))[0]
            new_annot_mat[updt_idx, idx] = 1.0
            idx += 1
    
    return new_names, new_annot_mat

# initialize log
def init_log(prefix):

    # get log file name
    log_file_name = prefix
    log_file_name += '.log'

    # create the log file
    log_format = '[%(levelname)s] %(message)s'
    logging.basicConfig(filename=log_file_name, filemode="w",
        level=logging.DEBUG, format=log_format)

    # add stderr as a stream handler
    stderr_handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(log_format)
    stderr_handler.setFormatter(formatter)
    logging.getLogger().addHandler(stderr_handler)

    # log time and command issued
    cur_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    logging.info('Command started at: %s' % cur_time)

# end the log
def end_log():
    cur_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    logging.info('Command finished at: %s' % cur_time)

# get command line
def get_command_line():
    
    parser = argparse.ArgumentParser(description='Get SVD of LD matrix')

    parser.add_argument('--bfile', dest='bfile', type=str, nargs=2,
        help='Plink file prefix', required=True)

    parser.add_argument('--annot', dest='annot', type=str, nargs='+',
        help='Annotation file prefix', required=True)

    parser.add_argument('--scale', dest='scale', type=float, default=1.0,
        help='Scale the theta', required=False)

    parser.add_argument('--hsq', dest='sim_hsq', type=float, nargs=2,
        help='Heritability to be simulated', required=True)

    parser.add_argument('--pct-causal-shared', dest='pct_causal_shared',
        type=float, default=0.08, help='Percentage of causal SNPs',
        required=False)

    parser.add_argument('--pct-causal-spec', dest='pct_causal_spec',
        type=float, default=0.02, help='Percentage of causal SNPs',
        required=False)

    parser.add_argument('--use-chrom', dest='use_chrom', type=int, nargs=2,
        help='Chromosomes to be used', required=True)

    parser.add_argument('--cont-annot', dest='cont_annot', type=str, nargs='*',
        help='A list of continuous annotations', required=False)

    parser.add_argument('--nbin', dest='nbin', type=int, default=5,
        help='Number of bins to split continuous annotations', required=False)

    parser.add_argument('--coef', dest='sim_coef_file', type=str,
        help='Coefficient file', required=True)

    parser.add_argument('--standardize', dest='standardize',
        help='Specify whether to standardize the data', required=False,
        action='store_true', default=False)

    parser.add_argument('--nsample', dest='nsample', type=int, nargs=2,
        help='Sample size of GWAS in the two populations', required=True)

    parser.add_argument('--std-annot', dest='std_annot',
        help='Specify whether to standardize the annotation', required=False,
        action='store_true', default=False)

    parser.add_argument('--info-only', dest='info_only',
        help='Specify whether to save summary stats', required=False,
        action='store_true', default=False)

    parser.add_argument('--num-sim', dest='num_sim', type=int,
        help='Number of simulations', required=True)

    parser.add_argument('--out', dest='out', type=str,
        help='Output file', required=True)
    
    args = parser.parse_args()
    
    return args

# run main
if __name__ == "__main__":
    main()
