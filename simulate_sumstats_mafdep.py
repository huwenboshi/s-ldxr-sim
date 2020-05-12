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

    # load bstat annotation
    start_chrom, end_chrom = args.use_chrom   
    annot_snps, annot_names, annot_mat = load_annot([args.bstat],
        start_chrom, end_chrom)

    # bstat 5 annot
    annot_vec = annot_mat[:,0]
    cut = pd.qcut(annot_vec, 5, retbins=True)[1]
    low, high = cut[3], cut[3+1]
    bstat5_idx = np.where((annot_vec>low) & (annot_vec<=high))[0]

    # start log
    init_log(args.out)

    # load minor allele frequencies
    maf1, maf2 = load_maf(args.bfile[0], args.bfile[1], args.use_chrom)
    maf_all_pop = maf1.merge(maf2, on=['SNP'])
    maf_all_pop['MAF_MIN'] = np.fmin(maf_all_pop['MAF_x'],maf_all_pop['MAF_y'])
    maf_all_pop['MAF_MAX'] = np.fmax(maf_all_pop['MAF_x'],maf_all_pop['MAF_y'])

    # iterate through the number of simulations
    for i in xrange(args.num_sim):    
       
        out_info = open('{}{}.info'.format(args.out,i+1), 'w')

        Z1, Z2 = sim_sumstats(args.bfile, args.pct_causal, args.rare_var_en,
            args.sim_hsq, args.nsample, args.gcor, out_info, i+1, start_chrom,
            end_chrom, args.out, bstat5_idx)
        
        Z1.to_csv('{}{}_pop1.sumstats.gz'.format(args.out,i+1),
            compression='gzip', index=False, sep='\t')

        Z2.to_csv('{}{}_pop2.sumstats.gz'.format(args.out,i+1),
            compression='gzip', index=False, sep='\t')

        out_info.close()

    # end lot
    end_log()

# load minor allele frequency info
def load_maf(bfile1, bfile2, use_chrom):
   
    maf1_all_chr = pd.DataFrame()
    maf2_all_chr = pd.DataFrame()
    
    for i in range(use_chrom[0], use_chrom[1]+1):
        
        maf1_chr = pd.read_csv('{}{}.frq'.format(bfile1, i),
            delim_whitespace=True)
        maf1_all_chr = pd.concat([maf1_all_chr, maf1_chr], ignore_index=True)
        
        maf2_chr = pd.read_csv('{}{}.frq'.format(bfile2, i),
            delim_whitespace=True)
        maf2_all_chr = pd.concat([maf2_all_chr, maf2_chr], ignore_index=True)

    return maf1_all_chr, maf2_all_chr

# simulate effect sizes
def sim_beta(maf_all_pop, hsq1, hsq2, gcor, all_rare_bstat5_causal_index,
    prop_causal_rare, all_comp_causal_index, prop_causal_common, alpha=-0.38):
   
    # simulate per snp per allele variance
    sigma = maf_all_pop['MAF_MAX'] * (1 - maf_all_pop['MAF_MAX']) + 1e-8
    per_snp_var = np.power(sigma, alpha)

    # possible causal index
    all_causal_index = np.where(maf_all_pop['MAF_MIN']>0.001)[0]
    all_rare_causal_index = np.where((maf_all_pop['MAF_MIN']>0.001) &
                                     (maf_all_pop['MAF_MIN']<0.05))[0]
    all_common_causal_index = np.where(maf_all_pop['MAF_MIN']>=0.05)[0]
    
    maf5pct_index = np.where(maf_all_pop['MAF_MIN']>=0.05)[0]
    
    ncausal_rare = int(prop_causal_rare*all_rare_bstat5_causal_index.shape[0])
    ncausal_common = int(prop_causal_common * all_comp_causal_index.shape[0])
    
    causal_rare_index = np.random.choice(all_rare_bstat5_causal_index,
        size=ncausal_rare, replace=False)
    causal_common_index = np.random.choice(all_comp_causal_index,
        size=ncausal_common, replace=False)
    
    causal_index = np.concatenate((causal_rare_index, causal_common_index),
        axis=None)
    causal_index = np.sort(causal_index)
    
    ncausal_snp = causal_index.shape[0]

    # per snp variance and covariance at causal snps
    cau_per_snp_var = per_snp_var[causal_index]
    maf5pct_per_snp_var = \
        per_snp_var[np.intersect1d(maf5pct_index, causal_index)]
    
    factor1 = hsq1 / np.sum(np.square(maf5pct_per_snp_var))
    factor2 = hsq2 / np.sum(np.square(maf5pct_per_snp_var))
    cau_per_snp_var1 = cau_per_snp_var * factor1
    cau_per_snp_var2 = cau_per_snp_var * factor2
    cau_per_snp_cov = gcor * np.sqrt(cau_per_snp_var1 * cau_per_snp_var2)

    # simulate effect size
    S = cau_per_snp_var2 - np.square(cau_per_snp_cov) / cau_per_snp_var1
    S[S<0.0] = 0.0
    L11 = np.sqrt(cau_per_snp_var1)
    L12 = cau_per_snp_cov / np.sqrt(cau_per_snp_var1)
    L22 = np.sqrt(S)

    # draw from normal
    tmp1 = np.random.normal(size=ncausal_snp)
    tmp2 = np.random.normal(size=ncausal_snp)

    # multiply cholesky
    beta1_cau = L11 * tmp1
    beta2_cau = L12 * tmp1 + L22 * tmp2
  
    # update beta
    nsnp_all = maf_all_pop.shape[0]
    beta1 = np.zeros(nsnp_all)
    beta2 = np.zeros(nsnp_all)
    beta1[causal_index] = beta1_cau
    beta2[causal_index] = beta2_cau

    return beta1, beta2, causal_index

def sim_sumstats(bfile, pct_causal, rare_en, hsq, nsample, gcor, out_info, idx,
    start_chrom, end_chrom, out_nm, bstat5_idx):

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

    # load minor allele frequencies
    maf1, maf2 = load_maf(bfile[0], bfile[1], [start_chrom, end_chrom])
    maf_all_pop = maf1.merge(maf2, on=['SNP'])
    maf_all_pop['MAF_MIN'] = np.fmin(maf_all_pop['MAF_x'],maf_all_pop['MAF_y'])
    maf_all_pop['MAF_MAX'] = np.fmax(maf_all_pop['MAF_x'],maf_all_pop['MAF_y'])

    # get indices of causal snps
    all_idx = np.array(range(0, maf1.shape[0])).astype(int)
    all_causal_idx = np.where(maf_all_pop['MAF_MIN']>0.001)[0]
    
    all_comm_idx = np.where(maf_all_pop['MAF_MAX']>0.05)[0]
    all_rare_idx = np.where(maf_all_pop['MAF_MAX']<=0.05)[0]
   
    all_comm_causal_idx = np.intersect1d(all_causal_idx, all_comm_idx)
    all_rare_causal_idx = np.intersect1d(all_causal_idx, all_rare_idx)

    all_comm_bstat5_causal_idx = np.intersect1d(all_comm_causal_idx,bstat5_idx)
    all_rare_bstat5_causal_idx = np.intersect1d(all_rare_causal_idx,bstat5_idx)

    all_comp_causal_idx = np.delete(all_idx, all_rare_bstat5_causal_idx)
    all_comp_causal_idx = np.intersect1d(all_comp_causal_idx, all_causal_idx) 

    # simulate causal effect size
    pct_causal_common = pct_causal / (1.0 + rare_en)
    pct_causal_rare = rare_en * pct_causal / (1.0 + rare_en) 
    beta1, beta2, cau_idx = sim_beta(maf_all_pop, hsq[0],
        hsq[1], gcor, all_rare_bstat5_causal_idx, pct_causal_rare,
        all_comp_causal_idx, pct_causal_common)

    # assess common / rare
    all_comp_bstat5_idx = np.delete(all_idx, bstat5_idx)
    all_comp_bstat5_comm_idx = np.intersect1d(all_comp_bstat5_idx,all_comm_idx)
    all_comp_bstat5_rare_idx = np.intersect1d(all_comp_bstat5_idx,all_rare_idx)
    all_bstat5_comm_idx = np.intersect1d(all_comm_idx, bstat5_idx)
    all_bstat5_rare_idx = np.intersect1d(all_rare_idx, bstat5_idx)

    beta1_sq = np.square(beta1)
    beta2_sq = np.square(beta2)

    bstat5_comm_persnp = np.sum(beta1_sq[all_bstat5_comm_idx]) \
        / all_bstat5_comm_idx.shape[0]
    bstat5_rare_persnp = np.sum(beta1_sq[all_bstat5_rare_idx]) \
        / all_bstat5_rare_idx.shape[0]
    
    comp_comm_persnp = np.sum(beta1_sq[all_comp_bstat5_comm_idx]) \
        / all_comp_bstat5_comm_idx.shape[0]
    comp_rare_persnp = np.sum(beta1_sq[all_comp_bstat5_rare_idx]) \
        / all_comp_bstat5_rare_idx.shape[0]

    bstat5_cr_ratio = bstat5_rare_persnp / bstat5_comm_persnp
    comp_cr_ratio = comp_rare_persnp / comp_comm_persnp

    nsample1, nsample2 = nsample

    # save simulated info
    gw_gcor = np.corrcoef(beta1, beta2)[0,1]
    out_info.write('{}\n'.format(gw_gcor))
    out_info.write('{}\n'.format(bstat5_cr_ratio / comp_cr_ratio))

    # simulate phenotype and summary stats for population 1
    beta1 = beta1.astype(np.float32)
    pheno1 = sim_pheno(bfile1, nsample1, start_chrom, end_chrom,
        cau_idx, beta1, legend)
    zsc_maf_thres = 0.01

    logging.info('Phenotype for population 1 simulated')
   
    out1_idx = np.where(maf1['MAF']>zsc_maf_thres)[0]
    
    Z1 = sim_zsc(bfile1, nsample1, start_chrom, end_chrom, pheno1, legend,
        maf1['MAF'].values)
    Z1_df = legend.loc[out1_idx,:].copy().reset_index(drop=True)
    Z1_df['Z'] = Z1; Z1_df['N'] = pheno1.shape[0]
    Z1_df = Z1_df[['SNP', 'CHR', 'BP', 'A1', 'A2', 'Z', 'N']]
    Z1_df['CHR'] = Z1_df['CHR'].astype(np.int32)

    logging.info('Summary stats for population 1 simulated')

    # simulate phenotype and summary stats for population 2
    beta2 = beta2.astype(np.float32)
    pheno2 = sim_pheno(bfile2, nsample2, start_chrom, end_chrom,
        cau_idx, beta2, legend)
    
    logging.info('Phenotype for population 2 simulated')
   
    out2_idx = np.where(maf2['MAF']>zsc_maf_thres)[0]
    Z2 = sim_zsc(bfile2, nsample2, start_chrom, end_chrom, pheno2, legend,
        maf2['MAF'].values)
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

def sim_pheno(bfile, nsample, start_chrom, end_chrom, cau_idx, beta,
    legend, nblock=40):
    
    mask = np.zeros(beta.shape[0], dtype=bool)
    mask[cau_idx] = True

    fam = '{}{}.fam'.format(bfile, start_chrom)
    indvs = pd.read_table(fam, nrows=nsample, header=None,
        delim_whitespace=True)
    nindv = indvs.shape[0]
    pheno = np.zeros(nindv, dtype=np.float32)
    
    for i in xrange(start_chrom, end_chrom+1):
        
        snpdata = Bed('{}{}.bed'.format(bfile, i),  count_A1=False)
        nsnp = snpdata.sid_count
        blocks = create_block(0, nsnp-1, nblock)

        snp_idx = np.where(legend['CHR']==i)[0]
        beta_chrom = beta[snp_idx]
        mask_chrom = mask[snp_idx]
        
        for blk in blocks:
            mask_chrom_blk = mask_chrom[blk]
            use_idx = blk[mask_chrom_blk==True]
            
            snpdata_blk = snpdata[0:nindv,use_idx]
            snpdata_blk = snpdata_blk.read(dtype=np.float32).val
            snpdata_blk -= snpdata_blk.mean(axis=0)
            
            pheno += np.dot(snpdata_blk, beta_chrom[use_idx])
    
    sigma_e = np.sqrt(1.0-np.var(pheno))
    
    eps = np.random.normal(scale=sigma_e, size=nindv).astype(np.float32)
    pheno += eps

    return pheno

def sim_zsc(bfile, nsample, start_chrom, end_chrom, pheno, legend,
    freq, nblock=40):
    
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
            snpdata_blk = snpdata_blk.read(dtype=np.float32).val
            snpdata_blk -= snpdata_blk.mean(axis=0)
            sigmasq = snpdata_blk.var(axis=0)
            zsc_chrom[use_idx] = np.dot(snpdata_blk.T,pheno)
            zsc_chrom[use_idx] /= np.sqrt(nindv*sigmasq)
        
        zsc[snp_idx] = zsc_chrom

    return zsc[freq>zsc_maf_thres]

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

    parser.add_argument('--gcor', dest='gcor', type=float, default=1.0,
        help='Scale the gen cov', required=False)

    parser.add_argument('--hsq', dest='sim_hsq', type=float, nargs=2,
        help='Heritability to be simulated', required=True)

    parser.add_argument('--rare-var-en', dest='rare_var_en', type=float,
        help='Enrichment in probability of sampling from low-freq variants',
        required=True)

    parser.add_argument('--bstat', dest='bstat', type=str,
        help='Background selection statistics annotation',
        required=True)

    parser.add_argument('--pct-causal', dest='pct_causal', type=float,
        default=0.1, help='Percentage of causal SNPs', required=True)

    parser.add_argument('--use-chrom', dest='use_chrom', type=int, nargs=2,
        help='Chromosomes to be used', required=True)

    parser.add_argument('--nsample', dest='nsample', type=int, nargs=2,
        help='Sample size of GWAS in the two populations', required=True)

    parser.add_argument('--num-sim', dest='num_sim', type=int,
        help='Number of simulations', required=True)

    parser.add_argument('--out', dest='out', type=str,
        help='Output file', required=True)
    
    args = parser.parse_args()
    
    return args

# run main
if __name__ == "__main__":
    main()
