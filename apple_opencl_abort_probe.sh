#!/bin/bash
# =====================================================================
#  Apple OpenCL silent-abort reproducer
#
#  WHAT THIS DOES
#    Runs the same compute kernel through Apple's OpenCL and through Metal,
#    with the output buffer pre-filled with a sentinel so that work-items
#    which never ran can be identified exactly.  Under load macOS aborts the
#    Metal command buffer that backs a dispatch.  Metal reports this; Apple's
#    OpenCL layer appears to report success anyway.  This script measures
#    whether that happens on YOUR Mac.
#
#  WHAT IT COLLECTS
#    Machine model, chip, core counts, RAM, macOS version, GPU name, the
#    OpenCL device/driver strings, the probe results, and macOS log lines
#    matching ONLY the text "command buffer was aborted" during the run.
#    It does not read your files, your network, or the rest of your log.
#    Everything is written to one plain-text report you can read before
#    sending.
#
#  WHAT IT DOES NOT DO
#    It does not send anything by itself and contains no mail credentials.
#    At the end it offers to open a pre-filled draft in Mail.app with the
#    report attached, which you review and send yourself.  Decline and it
#    just prints the file path.
#
#  REQUIREMENTS  macOS on Apple silicon, Xcode command line tools (clang).
#  USAGE
#    bash apple_opencl_abort_probe.sh              full ladder, offers a draft
#    bash apple_opencl_abort_probe.sh --quick      one load level, ~1 minute
#    bash apple_opencl_abort_probe.sh --no-mail    never mention email
#    bash apple_opencl_abort_probe.sh --help
#
#  FOR PEER REVIEWERS
#    Use --quick --no-mail.  The script is self-contained: it needs no
#    repository, no network and no credentials, and prints a one-line verdict
#    saying whether the reported behaviour occurred on your machine.
# =====================================================================
set -u

REPORT_TO="a.nejeoui@gmail.com"          # edit if you were given another address
QUICK=0; NOMAIL=0
for a in "$@"; do
  case "$a" in
    --quick)   QUICK=1 ;;
    --no-mail) NOMAIL=1 ;;
    --help|-h) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) printf 'unknown option: %s (try --help)\n' "$a"; exit 2 ;;
  esac
done
WORK="$(mktemp -d /tmp/oclabort.XXXXXX)"
REPORT="$HOME/Desktop/opencl-abort-report-$(hostname -s)-$(date +%Y%m%d-%H%M%S).txt"
START_EPOCH=$(date +%s)

cleanup(){ rm -rf "$WORK"; }
trap cleanup EXIT

say(){ printf '%s\n' "$*"; }
out(){ printf '%s\n' "$*" >> "$REPORT"; }

say ""
say "Apple OpenCL silent-abort reproducer"
say "===================================="
say ""
say "This builds two small programs, runs them for a few minutes, and writes"
say "a report to:"
say "  $REPORT"
say ""
say "Nothing is sent anywhere unless you choose to at the end."
say ""

# ---------- preflight ----------
if [ "$(uname -s)" != "Darwin" ]; then say "This script is for macOS only."; exit 1; fi
if ! command -v clang >/dev/null 2>&1; then
  say "clang not found.  Install the command line tools with:"
  say "    xcode-select --install"
  exit 1
fi

# ---------- environment ----------
out "Apple OpenCL silent-abort report"
out "generated $(date -u '+%Y-%m-%dT%H:%M:%SZ')  (schema 1)"
out ""
out "== machine =="
out "model            : $(sysctl -n hw.model 2>/dev/null)"
out "chip             : $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
out "cpu cores        : $(sysctl -n hw.ncpu 2>/dev/null) ($(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo '?') perf + $(sysctl -n hw.perflevel1.physicalcpu 2>/dev/null || echo '?') eff)"
out "memory           : $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 )) GB"
out "arch             : $(uname -m)"
out ""
out "== os =="
out "macOS            : $(sw_vers -productVersion 2>/dev/null) ($(sw_vers -buildVersion 2>/dev/null))"
out "darwin           : $(uname -r)"
out ""
out "== gpu =="
system_profiler SPDisplaysDataType 2>/dev/null \
  | grep -E "Chipset Model|Total Number of Cores|Vendor|Metal" | sed 's/^ */gpu              : /' >> "$REPORT"
out "external display : $(system_profiler SPDisplaysDataType 2>/dev/null | grep -c 'Display Type' | tr -d ' ') display entries reported"
out ""

# ---------- sources ----------
cat > "$WORK/ocl.c" <<'OCLEOF'
/* OpenCL half of the matched pair. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <OpenCL/opencl.h>
#include <sys/time.h>
static double now_s(void){struct timeval t;gettimeofday(&t,0);return t.tv_sec+t.tv_usec/1e6;}
#define SENT 0xDEADBEEFu
static const char *SRC =
"__kernel void heavy(__global uint *out, __constant uint *iters)\n"
"{\n"
"    size_t gid = get_global_id(0);\n"
"    uint acc = (uint)gid * 2654435761u + 1u;\n"
"    for (uint i = 0; i < iters[0]; i++) {\n"
"        acc = acc * 1664525u + 1013904223u;\n"
"        acc ^= acc >> 13;\n"
"        acc = acc * 2246822519u;\n"
"    }\n"
"    out[gid] = acc;\n"
"}\n";
int main(int argc,char**argv){
    unsigned n = argc>1?(unsigned)atoi(argv[1]):20000;
    unsigned iters = argc>2?(unsigned)atoi(argv[2]):100000000u;
    int runs = argc>3?atoi(argv[3]):3;
    cl_platform_id pl[8];cl_uint np=0;clGetPlatformIDs(8,pl,&np);
    cl_device_id dev=NULL;
    for(cl_uint i=0;i<np&&!dev;i++) clGetDeviceIDs(pl[i],CL_DEVICE_TYPE_GPU,1,&dev,NULL);
    if(!dev){printf("OPENCL_UNAVAILABLE\n");return 3;}
    char dn[256]={0},dv[256]={0},drv[256]={0};
    clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof dn,dn,NULL);
    clGetDeviceInfo(dev,CL_DEVICE_VERSION,sizeof dv,dv,NULL);
    clGetDeviceInfo(dev,CL_DRIVER_VERSION,sizeof drv,drv,NULL);
    cl_uint cu=0; clGetDeviceInfo(dev,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof cu,&cu,NULL);
    printf("opencl device    : %s\n",dn);
    printf("opencl version   : %s\n",dv);
    printf("opencl driver    : %s\n",drv);
    printf("opencl cu        : %u\n",cu);
    cl_int e;
    cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e);
    cl_command_queue q=clCreateCommandQueue(ctx,dev,0,&e);
    size_t sl=strlen(SRC);
    cl_program pr=clCreateProgramWithSource(ctx,1,&SRC,&sl,&e);
    if(clBuildProgram(pr,1,&dev,"",NULL,NULL)!=CL_SUCCESS){printf("OPENCL_BUILD_FAILED\n");return 3;}
    cl_kernel k=clCreateKernel(pr,"heavy",&e);
    size_t ob=(size_t)n*4;
    unsigned *h=malloc(ob);
    cl_mem dO=clCreateBuffer(ctx,CL_MEM_READ_WRITE,ob,NULL,&e);
    cl_mem dI=clCreateBuffer(ctx,CL_MEM_READ_ONLY,4,NULL,&e);
    clEnqueueWriteBuffer(q,dI,CL_TRUE,0,4,&iters,0,0,0);
    clSetKernelArg(k,0,sizeof(cl_mem),&dO); clSetKernelArg(k,1,sizeof(cl_mem),&dI);
    for(int r=0;r<runs;r++){
        for(unsigned i=0;i<n;i++) h[i]=SENT;
        clEnqueueWriteBuffer(q,dO,CL_TRUE,0,ob,h,0,0,0);
        size_t g=n; cl_event ev;
        double t0=now_s();
        cl_int eq=clEnqueueNDRangeKernel(q,k,1,NULL,&g,NULL,0,NULL,&ev);
        cl_int ef=clFinish(q);
        double dt=now_s()-t0;
        cl_int st=0; clGetEventInfo(ev,CL_EVENT_COMMAND_EXECUTION_STATUS,sizeof st,&st,NULL);
        clEnqueueReadBuffer(q,dO,CL_TRUE,0,ob,h,0,0,0);
        long un=0; for(unsigned i=0;i<n;i++) if(h[i]==SENT) un++;
        printf("  OpenCL run %d: %8.3fs enqueue=%d finish=%d event_status=%d unwritten=%ld/%u (%.1f%%)\n",
               r,dt,(int)eq,(int)ef,(int)st,un,n,100.0*un/n);
        clReleaseEvent(ev);
    }
    return 0;
}
OCLEOF

cat > "$WORK/mtl.m" <<'MTLEOF'
/* Metal half of the matched pair: identical kernel arithmetic. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/time.h>
static double now_s(void){struct timeval t;gettimeofday(&t,0);return t.tv_sec+t.tv_usec/1e6;}
#define SENT 0xDEADBEEFu
static NSString *SRC = @"#include <metal_stdlib>\n"
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
int main(int argc,char**argv){@autoreleasepool{
    uint32_t n=argc>1?(uint32_t)atoi(argv[1]):20000;
    uint32_t iters=argc>2?(uint32_t)atoi(argv[2]):100000000u;
    int runs=argc>3?atoi(argv[3]):3;
    id<MTLDevice> d=MTLCreateSystemDefaultDevice();
    if(!d){printf("METAL_UNAVAILABLE\n");return 3;}
    printf("metal device     : %s\n",[d.name UTF8String]);
    NSError *e=nil;
    id<MTLLibrary> lib=[d newLibraryWithSource:SRC options:nil error:&e];
    if(!lib){printf("METAL_BUILD_FAILED\n");return 3;}
    id<MTLComputePipelineState> ps=[d newComputePipelineStateWithFunction:[lib newFunctionWithName:@"heavy"] error:&e];
    id<MTLCommandQueue> q=[d newCommandQueue];
    id<MTLBuffer> ob=[d newBufferWithLength:(NSUInteger)n*4 options:MTLResourceStorageModeShared];
    id<MTLBuffer> ib=[d newBufferWithBytes:&iters length:4 options:MTLResourceStorageModeShared];
    NSUInteger tg=ps.maxTotalThreadsPerThreadgroup; if(tg>256)tg=256;
    for(int r=0;r<runs;r++){
        uint32_t *p=(uint32_t*)ob.contents;
        for(uint32_t i=0;i<n;i++)p[i]=SENT;
        id<MTLCommandBuffer> cb=[q commandBuffer];
        id<MTLComputeCommandEncoder> en=[cb computeCommandEncoder];
        [en setComputePipelineState:ps];[en setBuffer:ob offset:0 atIndex:0];[en setBuffer:ib offset:0 atIndex:1];
        [en dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(tg,1,1)];
        [en endEncoding];
        double t0=now_s();[cb commit];[cb waitUntilCompleted];double dt=now_s()-t0;
        long un=0;for(uint32_t i=0;i<n;i++)if(p[i]==SENT)un++;
        const char*st=cb.status==MTLCommandBufferStatusCompleted?"Completed":
                      cb.status==MTLCommandBufferStatusError?"Error":"other";
        printf("  Metal  run %d: %8.3fs status=%-9s error=%s unwritten=%ld/%u (%.1f%%)\n",
               r,dt,st,cb.error?[[cb.error localizedDescription]UTF8String]:"(nil)",un,n,100.0*un/n);
    }
    return 0;
}}
MTLEOF

# ---------- build ----------
say "Building..."
CLERR=0; MTERR=0
clang -O2 -std=c99 -DCL_TARGET_OPENCL_VERSION=120 "$WORK/ocl.c" -o "$WORK/ocl" -framework OpenCL 2>"$WORK/ocl.build" || CLERR=1
clang -fobjc-arc -O2 "$WORK/mtl.m" -o "$WORK/mtl" -framework Metal -framework Foundation 2>"$WORK/mtl.build" || MTERR=1
out "== build =="
out "opencl probe     : $([ $CLERR -eq 0 ] && echo ok || echo FAILED)"
out "metal probe      : $([ $MTERR -eq 0 ] && echo ok || echo FAILED)"
[ $CLERR -ne 0 ] && { out ""; out "-- opencl build log --"; sed 's/^/  /' "$WORK/ocl.build" >> "$REPORT"; }
[ $MTERR -ne 0 ] && { out ""; out "-- metal build log --";  sed 's/^/  /' "$WORK/mtl.build"  >> "$REPORT"; }
out ""

if [ $CLERR -ne 0 ] && [ $MTERR -ne 0 ]; then
  say "Both probes failed to build; the report has the compiler output."
  exit 1
fi

# ---------- run ----------
out "== probe results =="
out "Same kernel arithmetic through both APIs.  'unwritten' counts work-items"
out "whose output still holds the 0xDEADBEEF sentinel, i.e. never ran."
out ""
if [ $QUICK -eq 1 ]; then
  say "Running the probes (quick mode, about a minute)."
else
  say "Running the probes.  This takes a few minutes; please leave the machine"
  say "alone and do not put it to sleep."
fi
say ""

if [ $QUICK -eq 1 ]; then LADDER="400000000"; else LADDER="1000000 20000000 100000000 400000000"; fi
for IT in $LADDER; do
  out "-- 20000 work-items, $IT iterations each --"
  if [ $CLERR -eq 0 ]; then "$WORK/ocl" 20000 "$IT" 3 2>&1 | sed 's/^/  /' >> "$REPORT"; fi
  if [ $MTERR -eq 0 ]; then "$WORK/mtl" 20000 "$IT" 3 2>&1 | sed 's/^/  /' >> "$REPORT"; fi
  out ""
  say "  ...$IT iterations done"
done

# ---------- logs ----------
MINS=$(( ( $(date +%s) - START_EPOCH ) / 60 + 2 ))
out "== macOS log, command-buffer aborts during this run =="
out "predicate: eventMessage CONTAINS \"command buffer was aborted\"  (last ${MINS}m)"
out ""
log show --last "${MINS}m" --predicate 'eventMessage CONTAINS "command buffer was aborted"' 2>/dev/null \
  | grep -v "^Timestamp" | grep -v "log run noninteractively" | tail -40 | sed 's/^/  /' >> "$REPORT" \
  || out "  (log unavailable)"
ABORTS=$(log show --last "${MINS}m" --predicate 'eventMessage CONTAINS "command buffer was aborted"' 2>/dev/null | grep -c "aborted")
[ -z "${ABORTS:-}" ] && ABORTS=0
out ""
out "abort log lines  : $ABORTS"
out ""

# ---------- verdict ----------
OCL_SILENT=$(grep -c "OpenCL run .*event_status=0 unwritten=[1-9]" "$REPORT" 2>/dev/null)
MTL_LOUD=$(grep -c "Metal  run .*status=Error" "$REPORT" 2>/dev/null)
[ -z "${OCL_SILENT:-}" ] && OCL_SILENT=0
[ -z "${MTL_LOUD:-}" ]   && MTL_LOUD=0
out "== verdict =="
out "OpenCL dispatches reporting success with unwritten output : $OCL_SILENT"
out "Metal dispatches reporting Error                          : $MTL_LOUD"
if [ "$OCL_SILENT" -gt 0 ] && [ "$MTL_LOUD" -gt 0 ]; then
  out "RESULT: REPRODUCED -- OpenCL reported success for dispatches that did not"
  out "        run, while Metal reported the abort on this same machine."
elif [ "$OCL_SILENT" -gt 0 ]; then
  out "RESULT: REPRODUCED (OpenCL side) -- OpenCL reported success for"
  out "        dispatches that did not run, which is the defect itself.  Metal"
  out "        happened not to abort during its own runs here, so the"
  out "        side-by-side contrast is weaker; re-running usually shows it."
elif [ "$MTL_LOUD" -gt 0 ]; then
  out "RESULT: NOT REPRODUCED -- Metal aborted but OpenCL did not lose work."
else
  out "RESULT: NOT TRIGGERED -- no aborts occurred.  This machine may tolerate"
  out "        the load, or a heavier ladder is needed.  Still a useful data point."
fi
out ""
out "-- end of report --"

say ""
say "======================================================================"
sed -n '/== verdict ==/,$p' "$REPORT"
say "======================================================================"
say ""
say "Full report written to:"
say "  $REPORT"
say ""
say "Please read it before sending -- it is plain text and contains only what"
say "is listed at the top of this script."
say ""
if [ $NOMAIL -eq 1 ]; then
  say "Done.  (--no-mail given, so nothing will be sent.)"
  exit 0
fi
printf 'Open a pre-filled draft in Mail.app to %s? [y/N] ' "$REPORT_TO"
read -r ANS
case "$ANS" in
  y|Y|yes|YES)
    osascript >/dev/null 2>&1 <<APPLESCRIPT
tell application "Mail"
  set m to make new outgoing message with properties {subject:"OpenCL abort report: $(sysctl -n hw.model 2>/dev/null) / macOS $(sw_vers -productVersion 2>/dev/null)", content:"Report attached. Generated by apple_opencl_abort_probe.sh" & return & return, visible:true}
  tell m
    make new to recipient at end of to recipients with properties {address:"$REPORT_TO"}
    tell content to make new attachment with properties {file name:(POSIX file "$REPORT")} at after last paragraph
  end tell
  activate
end tell
APPLESCRIPT
    if [ $? -eq 0 ]; then
      say "A draft is open in Mail.  Review it and send when you are happy."
    else
      say "Could not open Mail.  Please attach this file to an email yourself:"
      say "  $REPORT"
    fi
    ;;
  *)
    say "Nothing sent.  To share it, attach this file to an email to $REPORT_TO:"
    say "  $REPORT"
    ;;
esac
say ""
