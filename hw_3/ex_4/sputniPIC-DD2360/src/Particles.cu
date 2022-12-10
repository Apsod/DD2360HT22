#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

int get_size(int3 stride){
    return stride.x * stride.y * stride.z;
}

int get_ptr(int3 i, int3 stride){
    return i.x * stride.y * stride.z + i.y * stride.z + i.z;
}


void prep_grid(struct grid* grd, FPfield3* out)
{
    int ptr; 
    int X = grd->nxn; 
    int Y = grd->nyn; 
    int Z = grd->nzn; 
    for(int x=1; x<X; x++){
        for(int y=1; y<Y; y++){
            for(int z=1; z<Z; z++){
                ptr = 2 * (x * Y * Z + y * Z + z);
                out[ptr] = make_fpfield3(
                        grd->XN[x - 1][y][z],
                        grd->YN[x][y - 1][z],
                        grd->ZN[x][y][z - 1]
                        );
                out[ptr+1] = make_fpfield3(
                        grd->XN[x][y][z],
                        grd->YN[x][y][z],
                        grd->ZN[x][y][z]
                    );
            }
        }
    }
}

void inner_loop(
        //struct particles* part,
        FPpart3* pos,
        FPpart3* vel,
        struct EMfield* field,
        FPpart3* grid,
        int3 grid_stride,
        //struct grid* grd,
        double3 L,
        FPfield3 invd,
        FPfield invVOL,
        FPpart dt_sub_cycling,
        FPpart qomdt2,
        char3 periodic,
        int NiterMover,
        int pix)
{
    // auxiliary variables
    FPpart dto2 = .5*dt_sub_cycling;
    FPpart omdtsq, denom;
    FPpart3 vt; 
    int3 i; 
    int3 fi;
    int ptr; 

    FPfield3 E, B;

    // local (to the particle) electric and magnetic field
    FPfield3 El, Bl;
    
    // interpolation densities
    FPfield weight; 
    FPfield3 N[2]; 

    // intermediate particle position and velocity
    FPpart3 p_;
    FPpart3 v_;
    
    FPpart3 p;
    FPpart3 v; 
    
    p_ = pos[pix];


    p = p_;

    v = vel[pix];

    
    // start is always zero, and the original code can't handle wrapping when it isnt.
    //double3 start = make_double3(grd->xStart, grd->yStart, grd->zStart);

    // calculate the average velocity iteratively
    // THIS LOOP IS SEQUENTIAL
    for(int innter=0; innter < NiterMover; innter++){
        // interpolation G-->P
        i = 2 + make_int3(p * invd);

        // calculate weights
        ptr = get_ptr(i, grid_stride)*2;
        N[0] = p - grid[ptr];
        N[1] = grid[ptr + 1] - p;

        // set to zero local electric and magnetic field
        El = make_fpfield3(0.0,0.0,0.0); 
        Bl = make_fpfield3(0.0,0.0,0.0); 
        
        // THIS LOOP IS PARALLELIZABLE (but only 8 * 6 = 48 operations unrolled)
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++){
                    weight = N[ii].x * N[jj].y * N[kk].z * invVOL;
                    fi = i - make_int3(ii, jj, kk);
                    E = make_fpfield3(
                            field->Ex[fi.x][fi.y][fi.z],
                            field->Ey[fi.x][fi.y][fi.z],
                            field->Ez[fi.x][fi.y][fi.z]
                            );
                    B = make_fpfield3(
                            field->Bxn[fi.x][fi.y][fi.z],
                            field->Byn[fi.x][fi.y][fi.z],
                            field->Bzn[fi.x][fi.y][fi.z]
                            );
                    El += weight * E;
                    Bl += weight * B; 
                }
        
        // end interpolation
        omdtsq = qomdt2*qomdt2*(dot(Bl, Bl));
        denom = 1.0/(1.0 + omdtsq);
        // solve the position equation
        vt = v + qomdt2*El;
        // solve the velocity equation
        v_ = (vt + qomdt2 * (cross(vt, Bl) + qomdt2 * dot(vt, Bl) * Bl)) * denom;
        // update position
        p = p_ + v_ * dto2; 
    } // end of iteration
    
    v = 2.0 * v_ - v; 
    p = p_ + v_ * dt_sub_cycling;

    if (p.x > L.x){
        if (periodic.x){ // PERIODIC
            p.x = p.x - L.x;
        } else { // REFLECTING BC
            v.x = -v.x;
            p.x = 2*L.x - p.x;
        }
    }
                                                                
    if (p.x < 0){
        if (periodic.x){ // PERIODIC
           p.x = p.x + L.x;
        } else { // REFLECTING BC
            v.x = -v.x;
            p.x = -p.x;
        }
    }

    if (p.y > L.y){
        if (periodic.y){ // PERIODIC
            p.y = p.y - L.y;
        } else { // REFLECTING BC
            v.y = -v.y;
            p.y = 2*L.y - p.y;
        }
    }
                                                                
    if (p.y < 0){
        if (periodic.y){ // PERIODIC
           p.y = p.y + L.y;
        } else { // REFLECTING BC
            v.y = -v.y;
            p.y = -p.y;
        }
    }

    if (p.z > L.z){
        if (periodic.z){ // PERIODIC
            p.z = p.z - L.z;
        } else { // REFLECTING BC
            v.z = -v.z;
            p.z = 2*L.z - p.z;
        }
    }
                                                                
    if (p.z < 0){
        if (periodic.z){ // PERIODIC
           p.z = p.z + L.z;
        } else { // REFLECTING BC
            v.z = -v.z;
            p.z = -p.z;
        }
    }
    vel[pix] = v;
    pos[pix] = p; 
}


/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;

    FPpart3 *pos, *vel, *grid;

    pos = new FPpart3[part->nop]; 
    vel = new FPpart3[part->nop];
    int3 grid_stride = make_int3(grd->nxn, grd->nyn, grd->nzn);
    // std::cout << "*** ALLOCATING GRID. SIZE: " << get_size(grid_stride) * 2 * 4 * 3 << " Bytes.  ***" << std::endl; 
    grid = new FPpart3[get_size(grid_stride) * 2];

    // std::cout << "*** PREPPING GRID ***" << std::endl; 
    prep_grid(grd, grid); 

    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5 * dt_sub_cycling;
    FPpart qomdt2 = part->qom*dto2/param->c;
    char3 periodic = make_char3(param->PERIODICX, param->PERIODICY, param->PERIODICZ);
    int NiterMover = part->NiterMover;

    double3 L = make_double3(grd->Lx, grd->Ly, grd->Lz);
    FPfield3 invd = make_fpfield3(grd->invdx, grd->invdy, grd->invdz); 
    FPfield invVOL = grd->invVOL;


    // std::cout << "*** PREPPING POS & VEL ***" << std::endl; 
    for (int i=0; i<part->nop; ++i){
        pos[i] = make_fppart3(part->x[i], part->y[i], part->z[i]);
        vel[i] = make_fppart3(part->u[i], part->v[i], part->w[i]);
    }

    // std::cout << "*** STARTING ***" << std::endl; 
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            inner_loop(pos, vel, field, grid, grid_stride, L, invd, invVOL, dt_sub_cycling, qomdt2, periodic, NiterMover, i);
        }  // end of subcycling
    } // end of one particle

    for (int i=0; i<part->nop; ++i){
        part->x[i] = pos[i].x;
        part->y[i] = pos[i].y;
        part->z[i] = pos[i].z;
        part->u[i] = vel[i].x;
        part->v[i] = vel[i].y;
        part->w[i] = vel[i].z;
    }

    return(0); // exit succcesfully
} // end of the mover

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}

