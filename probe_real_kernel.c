/* In-situ probe: runs the *real* library kernel (mpaKernel_32bits_opt.cl) at
   2048-bit MODEXP, which is the workload the fault was found in.  The output
   buffer is pre-filled with a sentinel so work-items that never ran can be
   identified exactly, and the event's execution status is inspected alongside
   the enqueue and clFinish returns.

   Needs the library's kernel:  MPA_KERNEL_DIR=/path/to/MPA-OpenCl/src
   Build: cc -O2 -std=c99 -DCL_TARGET_OPENCL_VERSION=120 \
              probe_real_kernel.c -o probe_real_kernel -framework OpenCL
   Usage: ./probe_real_kernel [n] [runs] [heavy] [local_size] [chunk]
            n           work-items                       (default 20000)
            runs        repetitions in one context       (default 1)
            heavy       1 = e=2^2048-1, 0 = e=3          (default 1)
            local_size  0 lets the driver choose         (default 0)
            chunk       0 = one launch, else sub-buffer chunk size
   See README.md. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <sys/time.h>

#define T 64                      /* 2048-bit operands at 32-bit words */
static double now_s(void){struct timeval tv;gettimeofday(&tv,0);return tv.tv_sec+tv.tv_usec/1e6;}
#define SENT 0xDEADBEEFu

int main(int argc, char **argv)
{
    long   n     = (argc > 1) ? atol(argv[1]) : 20000;
    int    runs  = (argc > 2) ? atoi(argv[2]) : 1;   /* repetitions in one context */
    /* exponent weight: 0 = tiny (e=3, cheap), 1 = full (e=2^2048-1, heavy) */
    int    heavy = (argc > 3) ? atoi(argv[3]) : 1;
    long   lsz   = (argc > 4) ? atol(argv[4]) : 0;   /* 0 = let the driver choose */
    long   chunk = (argc > 5) ? atol(argv[5]) : 0;   /* 0 = one single launch */
    const char *dir = getenv("MPA_KERNEL_DIR"); if (!dir) dir = "src";

    char path[512]; snprintf(path, sizeof path, "%s/mpaKernel_32bits_opt.cl", dir);
    FILE *fp = fopen(path, "rb"); if (!fp) { perror(path); return 2; }
    static char src[1<<21]; size_t sl = fread(src, 1, sizeof src - 1, fp); src[sl]=0; fclose(fp);

    cl_platform_id pl[8]; cl_uint np; clGetPlatformIDs(8, pl, &np);
    cl_device_id dev = NULL;
    for (cl_uint i = 0; i < np && !dev; i++)
        clGetDeviceIDs(pl[i], CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (!dev) { fprintf(stderr, "no GPU\n"); return 2; }
    char dn[256]; clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof dn, dn, NULL);

    cl_int err;
    cl_context ctx = clCreateContext(NULL,1,&dev,NULL,NULL,&err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    const char *csrc = src;
    cl_program pr = clCreateProgramWithSource(ctx,1,&csrc,&sl,&err);
    char opts[256]; snprintf(opts,sizeof opts,
        "-I%s -DWORDLENGTH_T=%d -DMPA_REGACC=1 -DMPA_FUSED_CIOS=1 -DMPA_UNROLL=1", dir, T);
    if (clBuildProgram(pr,1,&dev,opts,NULL,NULL)!=CL_SUCCESS){
        size_t ln; clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,0,NULL,&ln);
        char *lg=malloc(ln); clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,ln,lg,NULL);
        fprintf(stderr,"build failed:\n%s\n",lg); return 2; }
    cl_kernel k = clCreateKernel(pr,"mpaKernel",&err);

    size_t inB=(size_t)n*T*4, outB=inB;
    uint32_t *hA=calloc((size_t)n*T,4), *hB=calloc((size_t)n*T,4), *hC=malloc(outB);
    /* a = 3, e = 2^2048-1, modulus = all-ones-ish odd value: heavy MODEXP */
    for (long j=0;j<n;j++){ hA[j*T+T-1]=3u;
        if (heavy) { for(int i=0;i<T;i++) hB[j*T+i]=0xFFFFFFFFu; }
        else       { hB[j*T+T-1]=3u; } }
    uint32_t hP[2*T]; for(int i=0;i<T;i++) hP[i]=0xFFFFFFFFu; hP[T-1]=0xFFFFFFC5u;
    for(int i=0;i<T;i++) hP[T+i]=0u; hP[2*T-1]=1u;
    cl_int ob[4]={11 /*MODEXP*/,32,T*32,(cl_int)1u};

    cl_mem dA=clCreateBuffer(ctx,CL_MEM_READ_ONLY,inB,NULL,&err);
    cl_mem dB=clCreateBuffer(ctx,CL_MEM_READ_ONLY,inB,NULL,&err);
    cl_mem dC=clCreateBuffer(ctx,CL_MEM_READ_WRITE,outB,NULL,&err);
    cl_mem dO=clCreateBuffer(ctx,CL_MEM_READ_ONLY,sizeof ob,NULL,&err);
    cl_mem dP=clCreateBuffer(ctx,CL_MEM_READ_ONLY,sizeof hP,NULL,&err);
    clSetKernelArg(k,0,sizeof(cl_mem),&dA); clSetKernelArg(k,1,sizeof(cl_mem),&dB);
    clSetKernelArg(k,2,sizeof(cl_mem),&dC); clSetKernelArg(k,3,sizeof(cl_mem),&dO);
    clSetKernelArg(k,4,sizeof(cl_mem),&dP);
    clEnqueueWriteBuffer(q,dA,CL_TRUE,0,inB,hA,0,0,0);
    clEnqueueWriteBuffer(q,dB,CL_TRUE,0,inB,hB,0,0,0);
    clEnqueueWriteBuffer(q,dO,CL_TRUE,0,sizeof ob,ob,0,0,0);
    clEnqueueWriteBuffer(q,dP,CL_TRUE,0,sizeof hP,hP,0,0,0);

    {   cl_ulong pmem=0, lmem=0, gmem=0, maxalloc=0;
        size_t kwg=0, mult=0, dwg=0;
        clGetKernelWorkGroupInfo(k,dev,CL_KERNEL_PRIVATE_MEM_SIZE,sizeof pmem,&pmem,NULL);
        clGetKernelWorkGroupInfo(k,dev,CL_KERNEL_WORK_GROUP_SIZE,sizeof kwg,&kwg,NULL);
        clGetKernelWorkGroupInfo(k,dev,CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,sizeof mult,&mult,NULL);
        clGetDeviceInfo(dev,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof dwg,&dwg,NULL);
        clGetDeviceInfo(dev,CL_DEVICE_LOCAL_MEM_SIZE,sizeof lmem,&lmem,NULL);
        clGetDeviceInfo(dev,CL_DEVICE_GLOBAL_MEM_SIZE,sizeof gmem,&gmem,NULL);
        clGetDeviceInfo(dev,CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof maxalloc,&maxalloc,NULL);
        printf("device: %s   n=%ld   T=%d (%d-bit)\n", dn, n, T, T*32);
        printf("  kernel private mem/work-item : %llu bytes\n", (unsigned long long)pmem);
        printf("  kernel max work-group size   : %zu  (device max %zu, preferred multiple %zu)\n", kwg, dwg, mult);
        printf("  device local mem             : %llu bytes\n", (unsigned long long)lmem);
        printf("  device global mem            : %.2f GB, max alloc %.2f GB\n",
               gmem/1073741824.0, maxalloc/1073741824.0);
        printf("  n x private mem              : %.1f MB\n", (double)n*pmem/1048576.0);
    }
    for (int it=0; it<runs; it++) {
        for (size_t i=0;i<outB/4;i++) hC[i]=SENT;
        clEnqueueWriteBuffer(q,dC,CL_TRUE,0,outB,hC,0,0,0);   /* sentinel-fill */

        size_t ls=(size_t)lsz; cl_event ev=NULL;
        const size_t *lp = lsz>0 ? &ls : NULL;
        cl_int eq=CL_SUCCESS, ef=CL_SUCCESS, st=0;
        double t0=now_s();
        if (chunk <= 0 || chunk >= n) {
            size_t g=(size_t)n;
            eq=clEnqueueNDRangeKernel(q,k,1,NULL,&g,lp,0,NULL,&ev);
            ef=clFinish(q);
            clGetEventInfo(ev,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof st,&st,NULL);
        } else {
            /* Chunked: each launch is its own self-consistent problem.  The
               kernel indexes g*T+i, so a chunk starting at item k works on a
               sub-buffer whose origin is k*T*4 bytes into the parent. */
            for (long off=0; off<n && eq==CL_SUCCESS && ef==CL_SUCCESS; off+=chunk) {
                long nc = (n-off < chunk) ? (n-off) : chunk;
                cl_buffer_region rA={ (size_t)off*T*4, (size_t)nc*T*4 };
                cl_int e2;
                cl_mem sA=clCreateSubBuffer(dA,CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,&rA,&e2);
                cl_mem sB=clCreateSubBuffer(dB,CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION,&rA,&e2);
                cl_mem sC=clCreateSubBuffer(dC,CL_MEM_READ_WRITE,CL_BUFFER_CREATE_TYPE_REGION,&rA,&e2);
                if (e2!=CL_SUCCESS){ fprintf(stderr,"subbuffer failed %d\n",(int)e2); return 2; }
                clSetKernelArg(k,0,sizeof(cl_mem),&sA);
                clSetKernelArg(k,1,sizeof(cl_mem),&sB);
                clSetKernelArg(k,2,sizeof(cl_mem),&sC);
                size_t g=(size_t)nc;
                if (ev) clReleaseEvent(ev);
                eq=clEnqueueNDRangeKernel(q,k,1,NULL,&g,lp,0,NULL,&ev);
                ef=clFinish(q);
                clGetEventInfo(ev,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof st,&st,NULL);
                clReleaseMemObject(sA); clReleaseMemObject(sB); clReleaseMemObject(sC);
            }
            clSetKernelArg(k,0,sizeof(cl_mem),&dA);
            clSetKernelArg(k,1,sizeof(cl_mem),&dB);
            clSetKernelArg(k,2,sizeof(cl_mem),&dC);
        }
        double dt=now_s()-t0;
        clEnqueueReadBuffer(q,dC,CL_TRUE,0,outB,hC,0,0,0);

        long untouched=0;
        for (long j=0;j<n;j++){ int all=1;
            for(int i=0;i<T;i++) if(hC[j*T+i]!=SENT){all=0;break;}
            if(all) untouched++; }
        printf("  run %d: heavy=%d lsz=%ld chunk=%ld %8.3fs  enqueue=%d finish=%d event_status=%d"
               "  untouched_items=%ld/%ld (%.1f%%)\n",
               it, heavy, lsz, chunk, dt, (int)eq, (int)ef, (int)st, untouched, n, 100.0*untouched/n);
        if (ev) clReleaseEvent(ev);
    }
    return 0;
}
