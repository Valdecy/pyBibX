import numpy as np
cimport numpy as cnp

# returns for each string in u_ent the number of occurencences in the list of strin gs ent.
# optionally does not count occurrences but rather adds the number provided by acc 
cpdef get_counts_fast(list u_ent, list ent, cnp.int64_t[:] acc):
    cdef list counts
    cdef int i, j
    cdef int ents
    counts = []

    if (len(acc) == 0):
        for i in range(0, len(u_ent)):
            ents = 0
            for j in range(0, len(ent)):
                if (u_ent[i] in ent[j]):
                    ents = ents + 1
            counts.append(ents)
    elif (len(acc) > 0):
        for i in range(0, len(u_ent)):
            ents = 0
            for j in range(0, len(ent)):
                if (u_ent[i] in ent[j]):
                    ents = ents + acc[j]
            counts.append(ents)
    
    return counts


cpdef total_and_self_citations_fast(list u_aut, list aut, list citation, list ref):
    cdef str researcher
    cdef int cit, i1, i2
    t_c = []
    s_c = []
    for researcher in u_aut:
        doc = []
        cit = 0
        i1  = 0
        i2  = 0

        for researchers in aut:
            if (researcher in researchers):
                doc.append(citation[i1])
                for reference in ref[i2]:
                    if (researcher in reference.lower()):
                        cit = cit + 1
            i1 = i1 + 1
        i2 = i2 + 1
        t_c.append(sum(doc))
        s_c.append(cit)
    return t_c, s_c

cpdef h_index_fast(list u_aut, list aut, list citation, list ref):
    h_i = []
    cdef int i, j

    for researcher in u_aut:
        doc = []
        i   = 0
        for researchers in aut:
            if (researcher in researchers):
                doc.append(citation[i])
            i = i + 1
        for j in range(len(doc)-1, -1, -1):
            count = len([element for element in doc if element >= j])
            if (count >= j):
                h_i.append(j)
                break
    return h_i