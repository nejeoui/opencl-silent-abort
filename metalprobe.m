/* Does Metal report the abort that Apple's OpenCL layer hides?
   Same design as the OpenCL probe: sentinel-filled output, heavy per-thread
   work, then inspect the command buffer's status and error.

   Build: clang -fobjc-arc -O2 metalprobe.m -o metalprobe \
              -framework Metal -framework Foundation
   Usage: ./metalprobe [n] [iters] [runs]        (defaults 20000 1e8 3)

   The abort is probabilistic. If three runs all report Completed, raise
   `iters` or `runs`, and drive the display while it runs. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/time.h>

static double now_s(void){struct timeval tv;gettimeofday(&tv,0);return tv.tv_sec+tv.tv_usec/1e6;}
#define SENT 0xDEADBEEFu

static NSString *kSrc = @"#include <metal_stdlib>\n"
"using namespace metal;\n"
"kernel void heavy(device uint *out [[buffer(0)]],\n"
"                  constant uint &iters [[buffer(1)]],\n"
"                  uint gid [[thread_position_in_grid]])\n"
"{\n"
"    uint acc = gid * 2654435761u + 1u;\n"
"    for (uint i = 0; i < iters; i++) {\n"
"        acc = acc * 1664525u + 1013904223u;\n"
"        acc ^= acc >> 13;\n"
"        acc = acc * 2246822519u;\n"
"    }\n"
"    out[gid] = acc;\n"
"}\n";

int main(int argc, char **argv)
{
    @autoreleasepool {
        /* Defaults must be heavy enough to actually provoke the abort: at
           1e8 iterations roughly one run in three is aborted on an M2, so a
           single light run would print "Completed" and suggest, wrongly, that
           there is nothing to see. */
        uint32_t n     = (argc > 1) ? (uint32_t)atoi(argv[1]) : 20000;
        uint32_t iters = (argc > 2) ? (uint32_t)atoi(argv[2]) : 100000000u;
        int      runs  = (argc > 3) ? atoi(argv[3]) : 3;

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }
        printf("device: %s   n=%u   iters=%u\n", [dev.name UTF8String], n, iters);

        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:kSrc options:nil error:&err];
        if (!lib) { fprintf(stderr, "library: %s\n", [[err description] UTF8String]); return 2; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"heavy"];
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!ps) { fprintf(stderr, "pipeline: %s\n", [[err description] UTF8String]); return 2; }
        id<MTLCommandQueue> q = [dev newCommandQueue];

        id<MTLBuffer> outBuf = [dev newBufferWithLength:(NSUInteger)n*4
                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> itBuf  = [dev newBufferWithBytes:&iters length:4
                                   options:MTLResourceStorageModeShared];

        NSUInteger tg = ps.maxTotalThreadsPerThreadgroup;
        if (tg > 256) tg = 256;

        for (int r = 0; r < runs; r++) {
            uint32_t *p = (uint32_t *)outBuf.contents;
            for (uint32_t i = 0; i < n; i++) p[i] = SENT;

            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            [enc setBuffer:outBuf offset:0 atIndex:0];
            [enc setBuffer:itBuf  offset:0 atIndex:1];
            [enc dispatchThreads:MTLSizeMake(n,1,1)
                  threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
            [enc endEncoding];

            double t0 = now_s();
            [cb commit];
            [cb waitUntilCompleted];
            double dt = now_s() - t0;

            long untouched = 0;
            for (uint32_t i = 0; i < n; i++) if (p[i] == SENT) untouched++;

            const char *st = "?";
            switch (cb.status) {
                case MTLCommandBufferStatusNotEnqueued: st="NotEnqueued"; break;
                case MTLCommandBufferStatusEnqueued:    st="Enqueued";    break;
                case MTLCommandBufferStatusCommitted:   st="Committed";   break;
                case MTLCommandBufferStatusScheduled:   st="Scheduled";   break;
                case MTLCommandBufferStatusCompleted:   st="Completed";   break;
                case MTLCommandBufferStatusError:       st="Error";       break;
            }
            printf("  run %d: %7.3fs  status=%-11s error=%s  untouched=%ld/%u (%.1f%%)\n",
                   r, dt, st,
                   cb.error ? [[cb.error localizedDescription] UTF8String] : "(nil)",
                   untouched, n, 100.0*untouched/n);
            if (cb.error)
                printf("          domain=%s code=%ld\n",
                       [cb.error.domain UTF8String], (long)cb.error.code);
        }
    }
    return 0;
}
