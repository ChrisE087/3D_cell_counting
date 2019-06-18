import numpy as np
import nrrd

def overlap_mean(overlap_a, overlap_b):
    return overlap_a + overlap_b / 2
###############################################################################
# Generate test matrices that simulate eight overlapping patches 
###############################################################################
# Front upper left
ful = np.zeros((8,8,8))
ful = ful+1

# Front upper right
fur = np.zeros((8,8,8))
fur = fur+2

# Front down left
fdl = np.zeros((8,8,8))
fdl = fdl+3

# Front down right
fdr = np.zeros((8,8,8))
fdr = fdr+4

# Back upper left
bul = np.zeros((8,8,8))
bul = bul+5

# Back upper right
bur = np.zeros((8,8,8))
bur = bur+6

# Back down left
bdl = np.zeros((8,8,8))
bdl = bdl+7

# Back down right
bdr = np.zeros((8,8,8))
bdr = bdr+8

###############################################################################
# Get the overlappings
###############################################################################
# Specify the overlapping parts in each dimension
overlap_d0 = 4
overlap_d1 = 4
overlap_d2 = 4

# Get the front upper left overlappings
o_ful = ful.copy()
o_ful[0:overlap_d0, 0:overlap_d1, 0:overlap_d2] = 0

# Get the front upper right overlappings
o_fur = fur.copy()
o_fur[0:overlap_d0, overlap_d1:, 0:overlap_d2] = 0

# Get the front down left overlappings
o_fdl = fdl.copy()
o_fdl[overlap_d0:, 0:overlap_d1, 0:overlap_d2] = 0

# Get the front down right overlappings
o_fdr = fdr.copy()
o_fdr[overlap_d0:, overlap_d1:, 0:overlap_d2] = 0

# Get the back upper left overlappings
o_bul = bul.copy()
o_bul[0:overlap_d0, 0:overlap_d1, overlap_d2:] = 0

# Get the back upper right overlappings
o_bur = bur.copy()
o_bur[0:overlap_d0, overlap_d1:, overlap_d2:] = 0

# Get the back down left overlappings
o_bdl = bdl.copy()
o_bdl[overlap_d0:, 0:overlap_d1, overlap_d2:] = 0

# Get the back down right overlappings
o_bdr = bdr.copy()
o_bdr[overlap_d0:, overlap_d1:, overlap_d2:] = 0

###############################################################################
# Average the overlappings
###############################################################################
o_average = (o_ful + o_fur + o_fdl + o_fdr + o_bul + o_bur + o_bdl + o_bdr)/8

###############################################################################
# Overlay the non overlapping parts
###############################################################################
# Front upper left
o_average[0:overlap_d0, 0:overlap_d1, 0:overlap_d2] = ful[0:overlap_d0, 0:overlap_d1, 0:overlap_d2]

# Front upper right
o_average[0:overlap_d0, overlap_d1:, 0:overlap_d2] = fur[0:overlap_d0, overlap_d1:, 0:overlap_d2]

# Front down left
o_average[overlap_d0:, 0:overlap_d1, 0:overlap_d2] = fdl[overlap_d0:, 0:overlap_d1, 0:overlap_d2]

# Front down right
o_average[overlap_d0:, overlap_d1:, 0:overlap_d2] = fdr[overlap_d0:, overlap_d1:, 0:overlap_d2]

# Back upper left
o_average[0:overlap_d0, 0:overlap_d1, overlap_d2:] = bul[0:overlap_d0, 0:overlap_d1, overlap_d2:]

# Back upper right
o_average[0:overlap_d0, overlap_d1:, overlap_d2:] = bur[0:overlap_d0, overlap_d1:, overlap_d2:]

# Back down left
o_average[overlap_d0:, 0:overlap_d1, overlap_d2:] = bdl[overlap_d0:, 0:overlap_d1, overlap_d2:]

# Back down right
o_average[overlap_d0:, overlap_d1:, overlap_d2:] = bdr[overlap_d0:, overlap_d1:, overlap_d2:]

nrrd.write('oful.nrrd', o_ful)

# Calculate the mean of the overlappings
#ab = (a+b)/2 
