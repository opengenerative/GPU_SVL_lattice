

#include "VulkanApp.h"
#include "linearmath.h"
#include "MarchingCubes_kernel.h"
#include "Gratings.h"
#include "Fft_lattice.h"
#include <cufft.h>  
#include <helper_cuda.h>
#include "File_output.h"

#ifndef NDEBUG
#define ENABLE_VALIDATION (false)
#endif


using namespace std;

typedef float4 vect4;
typedef float2 vec2;

// tables
uint *d_numVertsTable = 0;

uint *d_edgeTable = 0;

uint *d_triTable = 0;


cudaPitchedPtr devPitchedPtr;
size_t tPitch = 0;
size_t slicepitch =0;
cudaExtent extend;

cufftHandle planr2c;
cufftHandle planc2r;

static float2 *fft_data = NULL;

static float2 *fft_data_compute = NULL;

static float2 *fft_data_compute_fill = NULL;

static float *fft_gratings = NULL;

static float2 *lattice_data = NULL;


class Mutlitopo : public VulkanBaseApp
{

    typedef struct UniformBufferObject_st {
        mat4x4 modelViewProj[5];
    } UniformBufferObject;

    struct LightPushConstants{
    fvec4 eyes;
    } push_constant;

    VkBuffer 
    //UC_cufftshift
    latticeonevol,
    //img_grating
    latticetwovol,
    latticethreevol,
    //img_grating_mesh
    vpos_one,vnorone,vpos_two,vnortwo,
    ///xyzone - xu,yu,zu
    xyzBufferone, 
    //X,Y,Z grid
    xyzBuffertwo,
    //X2,Y2,Z2 grid
    xyzBufferthree,
    indexBufferone,indexBuffertwo,indexBufferthree;
    
    VkDeviceMemory latticeoneMemory,latticeoneVolMemory, 
    vposMemory_one,vnormMemory_one,
    vposMemory_two,vnormMemory_two,
    latticetwoVolMemory,
    latticethreeVolMemory,
    xyzMemoryone,xyzMemorytwo,xyzMemorythree,
    indexMemoryone,indexMemorytwo,indexMemorythree;
    
    UniformBufferObject ubo;
  
    VkSemaphore vkWaitSemaphore, vkSignalSemaphore;
    MarchingCubeCuda mcalgo;

    Gratings lattice;
    Fft_lattice fftlattice;
    File_output output_file;
 

    cudaExternalSemaphore_t cudaWaitSemaphore, cudaSignalSemaphore, cudaTimelineSemaphore;
    cudaExternalMemory_t 
    cudaVertMemone,cudaVertMemtwo,
    cudaVertMemthree,
    cudaPosone, cudaPostwo,
    cudaNormone, cudaNormtwo;
  
    float *d_volumeone,*d_volumeone_one, *d_volumetwo, *d_volumethree, *d_volumethree_one;
    
    float4 *d_posone, *d_postwo,
    *d_normalone, *d_normaltwo;


    ///////////////for Topology ///////////////////
    uint Nxu;
    uint Nyu;
    uint Nzu;

    uint NumX ;
    uint NumY ;
    uint NumZ ;
    
    uint NumX2 ;
    uint NumY2 ;
    uint NumZ2 ;

    float isoValue;

    float dx2;
    float dy2;
    float dz2;

    char latticetype_one;
    char latticetype_two;

    bool ouput_data;

    const size_t maxmemvertsone;
    const size_t maxmemvertstwo;
    const uint size;
    const uint sizeone;
    const uint size2;

    int iter;
    float EndRes;

    int indi_range;
    int range_st ;

    size_t pitch_bytes;
    size_t grad_pitch_bytes;


    float *d_phi = NULL;
   
    float *d_theta = NULL;
    float *d_period = NULL;
  
    float2 *d_ga = NULL;
    float *d_svl = NULL;
  
    ///////////////////////////////////////////

    int FinalIter;
    float FinalRes;

    float Obj;
    float Obj_old;
    float Vol;

    ///view rotation
    int dist1;
    int dist2;
    int dist3;

    float angle1;
    float angle2;
    float angle3;

    bool shift;
 
    /////////////////////////////////////////
    //marching_cube
    uint3 gridSizeShiftone;
    uint3 gridSizeShifttwo;

    uint3 gridSizeone;
    uint3 gridSizetwo;
   
    uint3 gridSizeMaskone;
    uint3 gridSizeMasktwo;
 
    float3 voxelSizeone;
    float3 voxelSizetwo;

    float3 gridcenterone;
    float3 gridcentertwo;

    uint numVoxelsone ;
    uint numVoxelstwo ;

    uint activeVoxelsone;
    uint activeVoxelstwo;

    uint totalVertsone;
    uint totalVertstwo;

    bool g_bValidate = false;

    uint *d_voxelVertsone;
    uint *d_voxelVertstwo;
  
    uint *d_voxelVertsScanone;
    uint *d_voxelVertsScantwo;
 
    uint *d_voxelOccupiedone;
    uint *d_voxelOccupiedtwo;


  
    uint *d_voxelOccupiedScanone;
    uint *d_voxelOccupiedScantwo;

    uint *d_compVoxelArrayone;
    uint *d_compVoxelArraytwo;

    float dx1;
    float dy1;
    float dz1;

    float dx;
    float dy;
    float dz;
 
    float dxu;
  
    float mean_xu ;

    float dyu;
 
    float mean_yu;

    float dzu;

    float mean_zu;

public:
    Mutlitopo(size_t width, size_t height, size_t depth, float d_x, float d_y, float d_z, char a, char b, bool data_out) :
        VulkanBaseApp("LATTICE GENERATION USING GPU", ENABLE_VALIDATION),
        latticeonevol(VK_NULL_HANDLE),
        latticetwovol(VK_NULL_HANDLE),
        latticethreevol(VK_NULL_HANDLE),

        vpos_one(VK_NULL_HANDLE),
        vpos_two(VK_NULL_HANDLE),
        vnorone(VK_NULL_HANDLE),
        vnortwo(VK_NULL_HANDLE),

        xyzBufferone(VK_NULL_HANDLE),
        xyzBuffertwo(VK_NULL_HANDLE),
        xyzBufferthree(VK_NULL_HANDLE),
        indexBufferone(VK_NULL_HANDLE),
        indexBuffertwo(VK_NULL_HANDLE),

        latticeoneVolMemory(VK_NULL_HANDLE),
        latticetwoVolMemory(VK_NULL_HANDLE),
        latticethreeVolMemory(VK_NULL_HANDLE),
        
        vposMemory_one(VK_NULL_HANDLE),
        vposMemory_two(VK_NULL_HANDLE),
    
        vnormMemory_one(VK_NULL_HANDLE),
        vnormMemory_two(VK_NULL_HANDLE),
   
        xyzMemoryone(VK_NULL_HANDLE),
        xyzMemorytwo(VK_NULL_HANDLE),
        xyzMemorythree(VK_NULL_HANDLE),
      
        indexMemoryone(VK_NULL_HANDLE),
        indexMemorytwo(VK_NULL_HANDLE),
        indexMemorythree(VK_NULL_HANDLE),
    
        ubo(),
        mcalgo(),
        lattice(width,height,depth),
        fftlattice(),
        output_file(),

        vkWaitSemaphore(VK_NULL_HANDLE),
        vkSignalSemaphore(VK_NULL_HANDLE),
        cudaWaitSemaphore(),
        cudaSignalSemaphore(),
  
        cudaVertMemone(),
        cudaVertMemtwo(),
        cudaVertMemthree(),
     
        cudaPosone(),
        cudaPostwo(),
        cudaNormone(),
        cudaNormtwo(),

        d_posone(nullptr),
        d_postwo(nullptr),

        d_volumeone(nullptr),
        d_volumeone_one(nullptr),
        d_volumetwo(nullptr),
        d_volumethree(nullptr),
        d_volumethree_one(nullptr),
       
        d_normalone(nullptr),
        d_normaltwo(nullptr),
    
        Nxu(31),
        Nyu(31),
        Nzu(31),

        NumX(width),
        NumY(height),
        NumZ(depth),

        dx(d_x),
        dy(d_y),
        dz(d_z),

        latticetype_one(a),
        latticetype_two(b),

        ouput_data(data_out),

        dx2(0.5*d_x),
        dy2(0.5*d_y),
        dz2(0.5*d_z),

        NumX2(2*(NumX)),
        NumY2(2*(NumY)),
        NumZ2(2*(NumZ)),
        
        isoValue(0.5),
    
        size(NumX*NumY*NumZ),
        sizeone(Nxu*Nyu*Nzu),
        size2(NumX2*NumY2*NumZ2),

        maxmemvertsone(Nxu*Nyu*Nzu*4),
        maxmemvertstwo(max((NumX2*NumY2*NumZ2*4),300000)),
       
        EndRes(0.01),
        FinalIter(-1),
        FinalRes(-1.0),

        iter(500),
    
        ///view rotation
        dist1(MAX(Nzu,MAX(Nxu,Nyu))),
        dist2(MAX(NumZ,MAX(NumX,NumY))),
        dist3(MAX(NumZ2,MAX(NumX2,NumY2))),
        angle1(1.2),
        angle2(1.2),
        angle3(0.0),
        shift(false),
        ////////////////////////////////////////////////////////////////
        //marching_cube
        gridSizeShiftone(),
        gridSizeShifttwo(),

        gridSizeone(),
        gridSizetwo(),
    
        gridSizeMaskone(),
        gridSizeMasktwo(),
   
        voxelSizeone(),
        voxelSizetwo(),
   
        gridcenterone(),
        gridcentertwo(),

        numVoxelsone(0),
        numVoxelstwo(0),

        activeVoxelsone(0),
        activeVoxelstwo(0),

        totalVertsone(0),
        totalVertstwo(0),

        g_bValidate(false),

        d_voxelVertsone(nullptr),
        d_voxelVertstwo(nullptr),
  
        d_voxelVertsScanone(nullptr),
        d_voxelVertsScantwo(nullptr),

        d_voxelOccupiedone(nullptr),
        d_voxelOccupiedtwo(nullptr),
        
   
  
        d_voxelOccupiedScanone(nullptr),
        d_voxelOccupiedScantwo(nullptr),

        d_compVoxelArrayone(nullptr),
        d_compVoxelArraytwo(nullptr),
     
        mean_xu((Nxu-1)/2.0f),
        mean_yu((Nyu-1)/2.0f),
        mean_zu((Nzu-1)/2.0f),
        dxu(0.1),
        dyu(0.1),
        dzu(0.1),
        indi_range(3),
        range_st(floor(indi_range/2.0))
   
        {
           

            printf("\nNumX %d NumY %d NumZ %d NumX2 %d NumY2 %d NumZ2 %d dx %f dy %f dz %f  \n",NumX,NumY,NumZ,NumX2,NumY2,NumZ2,dx,dy,dz);
            /////Add compiled vulkan shader files
            char aone[] = "shaders/latticeone_grid.vert.spv";
            char atwo[] = "shaders/latticeone_grid.geom.spv";
            char athree[] = "shaders/latticeone_grid.frag.spv";
            
            char * vertex_shader_path = &aone[0];
            char * geometry_shader_path = &atwo[0];
            char * fragment_shader_path = &athree[0];
        
            shaderFiles_1.push_back(std::make_pair(VK_SHADER_STAGE_VERTEX_BIT, vertex_shader_path));
            shaderFiles_1.push_back(std::make_pair(VK_SHADER_STAGE_GEOMETRY_BIT, geometry_shader_path));
            shaderFiles_1.push_back(std::make_pair(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_shader_path));

            char mone[]   = "shaders/latticeone_mesh.vert.spv";
            char mtwo[]   = "shaders/latticeone_mesh.geom.spv";
            char mthree[] ="shaders/latticeone_mesh.frag.spv";
            
            char * bmesh_vertex_shader_path = &mone[0];
            char * bmesh_geometry_shader_path = &mtwo[0];
            char * bmesh_fragment_shader_path = &mthree[0];

            shaderFilesone.push_back(std::make_pair(VK_SHADER_STAGE_VERTEX_BIT, bmesh_vertex_shader_path));
            shaderFilesone.push_back(std::make_pair(VK_SHADER_STAGE_GEOMETRY_BIT, bmesh_geometry_shader_path));
            shaderFilesone.push_back(std::make_pair(VK_SHADER_STAGE_FRAGMENT_BIT, bmesh_fragment_shader_path));

            char bone[] = "shaders/latticetwo_grid.vert.spv";
            char btwo[] = "shaders/latticetwo_grid.geom.spv";
            char bthree[] ="shaders/latticetwo_grid.frag.spv";
            
            char * vertex_shader_path_two = &bone[0];
            char * geometry_shader_path_two = &btwo[0];
            char * fragment_shader_path_two = &bthree[0];
        
            shaderFiles_2.push_back(std::make_pair(VK_SHADER_STAGE_VERTEX_BIT, vertex_shader_path_two));
            shaderFiles_2.push_back(std::make_pair(VK_SHADER_STAGE_GEOMETRY_BIT, geometry_shader_path_two));
            shaderFiles_2.push_back(std::make_pair(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_shader_path_two));

            char ione[] = "shaders/latticetwo_mesh.vert.spv";
            char itwo[] = "shaders/latticetwo_mesh.geom.spv";
            char ithree[] ="shaders/latticetwo_mesh.frag.spv";
            
            char * vertex_shader_path_nine = &ione[0];
            char * geometry_shader_path_nine = &itwo[0];
            char * fragment_shader_path_nine = &ithree[0];
        
            shaderFilestwo.push_back(std::make_pair(VK_SHADER_STAGE_VERTEX_BIT, vertex_shader_path_nine));
            shaderFilestwo.push_back(std::make_pair(VK_SHADER_STAGE_GEOMETRY_BIT, geometry_shader_path_nine));
            shaderFilestwo.push_back(std::make_pair(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_shader_path_nine));

            char cone[] = "shaders/latticethree_grid.vert.spv";
            char ctwo[] = "shaders/latticethree_grid.geom.spv";
            char cthree[] ="shaders/latticethree_grid.frag.spv";
            
            char * vertex_shader_path_three = &cone[0];
            char * geometry_shader_path_three = &ctwo[0];
            char * fragment_shader_path_three = &cthree[0];
        
            shaderFiles_3.push_back(std::make_pair(VK_SHADER_STAGE_VERTEX_BIT, vertex_shader_path_three));
            shaderFiles_3.push_back(std::make_pair(VK_SHADER_STAGE_GEOMETRY_BIT, geometry_shader_path_three));
            shaderFiles_3.push_back(std::make_pair(VK_SHADER_STAGE_FRAGMENT_BIT, fragment_shader_path_three));


          
        }


    ~Mutlitopo() {
        
        if (vkSignalSemaphore != VK_NULL_HANDLE) {
            
            checkCudaErrors(cudaDestroyExternalSemaphore(cudaSignalSemaphore));
            vkDestroySemaphore(device, vkSignalSemaphore, nullptr);
        }
        if (vkWaitSemaphore != VK_NULL_HANDLE) {
            
            checkCudaErrors(cudaDestroyExternalSemaphore(cudaWaitSemaphore));
            vkDestroySemaphore(device, vkWaitSemaphore, nullptr);
        }

        if (xyzBufferone != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, xyzBufferone, nullptr);
        }

        if (xyzBuffertwo != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, xyzBuffertwo, nullptr);
        }

        if (xyzBufferthree != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, xyzBufferthree, nullptr);
        }

        if (xyzMemoryone != VK_NULL_HANDLE) {
            vkFreeMemory(device, xyzMemoryone, nullptr);
        }

        if (xyzMemorytwo != VK_NULL_HANDLE) {
            vkFreeMemory(device, xyzMemorytwo, nullptr);
        }

        if (xyzMemorythree != VK_NULL_HANDLE) {
            vkFreeMemory(device, xyzMemorythree, nullptr);
        }

        if (latticeonevol != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, latticeonevol, nullptr);
        }

        if (latticetwovol != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, latticetwovol, nullptr);
        }

        if (latticethreevol != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, latticethreevol, nullptr);
        }

        if (vpos_one != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, vpos_one, nullptr);
        }

        if (vpos_two != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, vpos_two, nullptr);
        }

        if (vnorone != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, vnorone, nullptr);
        }

        if (vnortwo != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, vnortwo, nullptr);
        }

        if (latticeoneVolMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, latticeoneVolMemory, nullptr);
        }

        if (latticetwoVolMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, latticetwoVolMemory, nullptr);
        }

        if (latticethreeVolMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, latticethreeVolMemory, nullptr);
        }

        if (vposMemory_one != VK_NULL_HANDLE) {
            vkFreeMemory(device, vposMemory_one, nullptr);
        }

        if (vnormMemory_one != VK_NULL_HANDLE) {
            vkFreeMemory(device, vnormMemory_one, nullptr);
        }

        if (vposMemory_two != VK_NULL_HANDLE) {
            vkFreeMemory(device, vposMemory_two, nullptr);
        }

        if (vnormMemory_two != VK_NULL_HANDLE) {
            vkFreeMemory(device, vnormMemory_two, nullptr);
        }

        if (d_volumeone) {
            checkCudaErrors(cudaDestroyExternalMemory(cudaVertMemone));
        }

        if (d_volumetwo) {
            checkCudaErrors(cudaDestroyExternalMemory(cudaVertMemtwo));
        }

        if (d_volumethree) {
            checkCudaErrors(cudaDestroyExternalMemory(cudaVertMemthree));
        }

        if (d_posone) {
           
            checkCudaErrors(cudaDestroyExternalMemory(cudaPosone));
        }

        if (d_postwo) {
           
            checkCudaErrors(cudaDestroyExternalMemory(cudaPostwo));
        }

        if (d_normalone) {
            checkCudaErrors(cudaDestroyExternalMemory(cudaNormone));
        }

        if (d_normaltwo) {
            checkCudaErrors(cudaDestroyExternalMemory(cudaNormtwo));
        }

        if (indexBufferone != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, indexBufferone, nullptr);
        }

        if (indexBuffertwo != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, indexBuffertwo, nullptr);
        }

        if (indexBufferthree != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, indexBufferthree, nullptr);
        }

        if (indexMemoryone != VK_NULL_HANDLE) {
            vkFreeMemory(device, indexMemoryone, nullptr);
        }

        if (indexMemorytwo != VK_NULL_HANDLE) {
            vkFreeMemory(device, indexMemorytwo, nullptr);
        }

        if (indexMemorythree != VK_NULL_HANDLE) {
            vkFreeMemory(device, indexMemorythree, nullptr);
        }

    }

    void getVertexDescriptions_1(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc) 
    {
        bindingDesc.resize(2);
        attribDesc.resize(2);

        bindingDesc[0].binding = 0;
        bindingDesc[0].stride = sizeof(float);
        bindingDesc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        bindingDesc[1].binding = 1;
        bindingDesc[1].stride = sizeof(vec3);
        bindingDesc[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        attribDesc[0].binding = 0;
        attribDesc[0].location = 0;
        attribDesc[0].format = VK_FORMAT_R32_SFLOAT;
        attribDesc[0].offset = 0;

        attribDesc[1].binding = 1;
        attribDesc[1].location = 1;
        attribDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribDesc[1].offset = 0;


    }


        
    void getVertexDescriptionsone(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc) {
        bindingDesc.resize(2);
        attribDesc.resize(2);

        bindingDesc[0].binding = 0;
        bindingDesc[0].stride = sizeof(float4);
        bindingDesc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        bindingDesc[1].binding = 1;
        bindingDesc[1].stride = sizeof(vec4);
        bindingDesc[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        attribDesc[0].binding = 0;
        attribDesc[0].location = 0;
        attribDesc[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attribDesc[0].offset = 0;

        attribDesc[1].binding = 1;
        attribDesc[1].location = 1;
        attribDesc[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attribDesc[1].offset = 0;

    }

    void getVertexDescriptions_2(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc) {
        
        bindingDesc.resize(2);
        attribDesc.resize(2);

        bindingDesc[0].binding = 0;
        bindingDesc[0].stride = sizeof(float);
        bindingDesc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        bindingDesc[1].binding = 1;
        bindingDesc[1].stride = sizeof(vec3);
        bindingDesc[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        attribDesc[0].binding = 0;
        attribDesc[0].location = 0;
        attribDesc[0].format = VK_FORMAT_R32_SFLOAT;
        attribDesc[0].offset = 0;

        attribDesc[1].binding = 1;
        attribDesc[1].location = 1;
        attribDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribDesc[1].offset = 0;

    }



    void getVertexDescriptionstwo(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc) 
    {
        bindingDesc.resize(2);
        attribDesc.resize(2);

        bindingDesc[0].binding = 0;
        bindingDesc[0].stride = sizeof(float4);
        bindingDesc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        bindingDesc[1].binding = 1;
        bindingDesc[1].stride = sizeof(vec4);
        bindingDesc[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        attribDesc[0].binding = 0;
        attribDesc[0].location = 0;
        attribDesc[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attribDesc[0].offset = 0;

        attribDesc[1].binding = 1;
        attribDesc[1].location = 1;
        attribDesc[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attribDesc[1].offset = 0;

    }


    

    void getVertexDescriptions_3(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc) 
    {
        bindingDesc.resize(2);
        attribDesc.resize(2);

        bindingDesc[0].binding = 0;
        bindingDesc[0].stride = sizeof(float);
        bindingDesc[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        bindingDesc[1].binding = 1;
        bindingDesc[1].stride = sizeof(vec3);
        bindingDesc[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        attribDesc[0].binding = 0;
        attribDesc[0].location = 0;
        attribDesc[0].format = VK_FORMAT_R32_SFLOAT;
        attribDesc[0].offset = 0;

        attribDesc[1].binding = 1;
        attribDesc[1].location = 1;
        attribDesc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribDesc[1].offset = 0;


    }


    void getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info) 
    {
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        info.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
        info.primitiveRestartEnable = VK_FALSE;
    }


    void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait, std::vector< VkPipelineStageFlags>& waitStages) const 
    {
        if (currentFrame != 0) {
            wait.push_back(vkWaitSemaphore);
            waitStages.push_back(VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        }
    }

    void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const 
    {
        signal.push_back(vkSignalSemaphore);
    }

    void initVulkanApp() {

        const size_t nVertsone = (Nxu)*(Nyu)*(Nzu);
        const size_t nVertstwo = (NumX)*(NumY)*(NumZ);
        const size_t nVertsthree = NumX2*NumY2*NumZ2;
       
        const size_t nIndsone =  (Nxu)*(Nxu)*(Nxu);
        const size_t nIndstwo =  (NumX)*(NumY)*(NumZ);
        const size_t nIndsthree =  (NumX2)*(NumY2)*(NumZ2);

        createExternalBuffer(nVertsone * sizeof(float),
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             getDefaultMemHandleType(),
                             latticeonevol, latticeoneVolMemory);

        createExternalBuffer(nVertstwo* sizeof(float),
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             getDefaultMemHandleType(),
                             latticetwovol, latticetwoVolMemory);

        createExternalBuffer(nVertsthree *sizeof(float),
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                             getDefaultMemHandleType(),
                             latticethreevol, latticethreeVolMemory);

        createExternalBuffer(maxmemvertsone * sizeof(float4),
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        getDefaultMemHandleType(),
                        vpos_one, vposMemory_one);

        createExternalBuffer(maxmemvertstwo * sizeof(float4),
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        getDefaultMemHandleType(),
                        vpos_two, vposMemory_two);

        createExternalBuffer(maxmemvertsone * sizeof(float4),
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            getDefaultMemHandleType(),
                            vnorone, vnormMemory_one);

        createExternalBuffer(maxmemvertstwo * sizeof(float4),
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                            getDefaultMemHandleType(),
                            vnortwo, vnormMemory_two);

        createBuffer(nVertsone * sizeof(vec3),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     xyzBufferone, xyzMemoryone);

        createBuffer(nVertstwo * sizeof(vec3),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     xyzBuffertwo, xyzMemorytwo);

        createBuffer(nVertsthree * sizeof(vec3),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     xyzBufferthree, xyzMemorythree);

        createBuffer(nIndsone * sizeof(uint32_t),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     indexBufferone, indexMemoryone);

        createBuffer(nIndstwo * sizeof(uint32_t),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     indexBuffertwo, indexMemorytwo);
        
        createBuffer(nIndsthree * sizeof(uint32_t),
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     indexBufferthree, indexMemorythree);
       
        importCudaExternalMemory((void **)&d_volumeone, cudaVertMemone, latticeoneVolMemory, nVertsone * sizeof(float), getDefaultMemHandleType());
        importCudaExternalMemory((void **)&d_volumetwo, cudaVertMemtwo, latticetwoVolMemory, nVertstwo * sizeof(float), getDefaultMemHandleType());
        importCudaExternalMemory((void **)&d_volumethree, cudaVertMemthree, latticethreeVolMemory, nVertsthree * sizeof(float), getDefaultMemHandleType());  
        
        importCudaExternalMemory((void **)&d_posone, cudaPosone,vposMemory_one, maxmemvertsone * sizeof(*d_posone), getDefaultMemHandleType());
        importCudaExternalMemory((void **)&d_postwo, cudaPostwo,vposMemory_two, maxmemvertstwo * sizeof(*d_postwo), getDefaultMemHandleType());
        
        importCudaExternalMemory((void **)&d_normalone, cudaNormone,vnormMemory_one, maxmemvertsone * sizeof(*d_normalone), getDefaultMemHandleType());
        importCudaExternalMemory((void **)&d_normaltwo, cudaNormtwo,vnormMemory_two, maxmemvertstwo * sizeof(*d_normaltwo), getDefaultMemHandleType());
       
        //////////////////////////////// 3d grid position /////////////////////////////////
        {
            // Set up the initial values for the vertex buffers with Vulkan
            void *stagingBase;
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;
            VkDeviceSize stagingSz = nVertsone * sizeof(vec3);
            createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

            vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);

            uint cou = 0;
            for (size_t z =0; z<Nzu; z++)
            {
                for (size_t y = 0; y < Nyu; y++) 
                {
                    for (size_t x = 0; x < Nxu; x++) 
                    {
                      
                        vec3 *stagedVert = (vec3 *)stagingBase;
                        stagedVert[cou][0] = x ;
                        stagedVert[cou][1] = y ;
                        stagedVert[cou][2] = z ;
                        cou++;
                    }
                }
            }

            copyBuffer(xyzBufferone, stagingBuffer,0, nVertsone * sizeof(vec3));
            vkUnmapMemory(device, stagingMemory);
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMemory, nullptr);
        }

        {
            // Set up the initial values for the vertex buffers with Vulkan
            void *stagingBase;
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;
            VkDeviceSize stagingSz = nVertstwo * sizeof(vec3);
            createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

            vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);

            uint cou = 0;
            for (size_t z =0; z<NumZ; z++)
            {
                for (size_t y = 0; y < NumY; y++) 
                {
                    for (size_t x = 0; x < NumX; x++) 
                    {
                      
                        vec3 *stagedVert = (vec3 *)stagingBase;
                        stagedVert[cou][0] = x;
                        stagedVert[cou][1] = y;
                        stagedVert[cou][2] = z;
                        cou++;
                    }
                }
            }

            copyBuffer(xyzBuffertwo, stagingBuffer,0, nVertstwo * sizeof(vec3));
            vkUnmapMemory(device, stagingMemory);
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMemory, nullptr);
        }


        {
            // Set up the initial values for the vertex buffers with Vulkan
            void *stagingBase;
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;
            VkDeviceSize stagingSz = nVertsthree * sizeof(vec3);
            createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

            vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);

            uint cou = 0;
            for (size_t z =0; z<NumZ2; z++)
            {
                for (size_t y = 0; y < NumY2; y++) 
                {
                    for (size_t x = 0; x < NumX2; x++) 
                    {
                      
                        vec3 *stagedVert = (vec3 *)stagingBase;
                        stagedVert[cou][0] = x * dx2 ;
                        stagedVert[cou][1] = y * dy2 ;
                        stagedVert[cou][2] = z * dz2 ;
                        cou++;
                    }
                }
            }

            copyBuffer(xyzBufferthree, stagingBuffer,0, nVertsthree * sizeof(vec3));
            vkUnmapMemory(device, stagingMemory);
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMemory, nullptr);
        }



        /////////////////////////// Indices ///////////////////////////////////////    
        {
            
            void *stagingBase;
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;
            VkDeviceSize stagingSz = nIndsone * sizeof(uint32_t);
            createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

            vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);
            
            
            uint32_t *indices = (uint32_t *)stagingBase;
            uint32_t cou = 0;
            for (size_t z = 0; z < Nzu; z++)
            {
                for (size_t y = 0; y < Nyu; y++) 
                {
                    for (size_t x = 0; x < Nxu; x++) 
                    {
                        
                        indices[cou] = cou;
                        cou++;
                    }
                }
            }
            

            copyBuffer(indexBufferone, stagingBuffer,0, nIndsone * sizeof(uint32_t));
            vkUnmapMemory(device, stagingMemory);
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMemory, nullptr);
            
        }


                /////////////////////////// Indices ///////////////////////////////////////    
        {
            
            void *stagingBase;
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;
            VkDeviceSize stagingSz = nIndstwo * sizeof(uint32_t);
            createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

            vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);
            
            
            uint32_t *indices = (uint32_t *)stagingBase;
            uint32_t cou = 0;
            for (size_t z = 0; z < NumZ; z++)
            {
                for (size_t y = 0; y < NumY; y++) 
                {
                    for (size_t x = 0; x < NumX; x++) 
                    {
                        
                        indices[cou] = cou;
                        cou++;
                    }
                }
            }
            

            copyBuffer(indexBuffertwo, stagingBuffer,0, nIndstwo * sizeof(uint32_t));
            vkUnmapMemory(device, stagingMemory);
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMemory, nullptr);
            
        }


                /////////////////////////// Indices ///////////////////////////////////////    
        {
            
            void *stagingBase;
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;
            VkDeviceSize stagingSz = nIndsthree * sizeof(uint32_t);
            createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

            vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);
            
            
            uint32_t *indices = (uint32_t *)stagingBase;
            uint32_t cou = 0;
            for (size_t z = 0; z < NumZ2; z++)
            {
                for (size_t y = 0; y < NumY2; y++) 
                {
                    for (size_t x = 0; x < NumX2; x++) 
                    {
                        
                        indices[cou] = cou;
                        cou++;
                    }
                }
            }
            

            copyBuffer(indexBufferthree, stagingBuffer,0, nIndsthree * sizeof(uint32_t));
            vkUnmapMemory(device, stagingMemory);
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMemory, nullptr);
            
        }



                /////////////////////////////////// float ////////////////////////////////////
        {
            void *stagingBase;
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;
            VkDeviceSize stagingSz = nVertsthree  * sizeof(float);
            createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

            vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);
            
            uint cou = 0;

            float *heightval = (float *)stagingBase;
            for (size_t z =0; z<NumZ2; z++)
            {
                for (size_t y = 0; y < NumY2; y++) 
                {
                    for (size_t x = 0; x < NumX2; x++) 
                    {
                        heightval[cou] = atan2(y,x);
                        cou++;
                    }
                }
            }
                  
            copyBuffer(latticethreevol, stagingBuffer,0, nVertsthree  * sizeof(float));
            vkUnmapMemory(device, stagingMemory);
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMemory, nullptr);
        }

        /////////////////////////////////// Theta ////////////////////////////////////
        {
            void *stagingBase;
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;
            VkDeviceSize stagingSz = nVertstwo* sizeof(float);
            createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

            vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);

            uint cou = 0;

            float *heightval = (float *)stagingBase;
        
            for (size_t z =0; z<NumZ; z++){
                for (size_t y = 0; y < NumY; y++) 
                {
                    for (size_t x = 0; x < NumX; x++) 
                    {

                        //heightval[cou] = sqrt(pow(x,2) + pow(y,2));
                        heightval[cou] = atan2(y+1,x+1);
                        cou++;
                    }
                }
            }
            
            copyBuffer(latticetwovol, stagingBuffer,0, nVertstwo * sizeof(float));
            vkUnmapMemory(device, stagingMemory);
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMemory, nullptr);

        }

        
        /////////////////////////////////// unitcell lattice vol ////////////////////////////////////
        {
            
            
            void *stagingBase;
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;
            VkDeviceSize stagingSz = nVertsone * sizeof(float);
            createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

            vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);
           
            float *heightval = (float *)stagingBase;
            uint cou = 0;
            for (size_t z =0; z<Nzu; z++)
            {
                for (size_t y = 0; y < Nyu; y++) 
                {
                    for (size_t x = 0; x < Nxu; x++) 
                    {
                        heightval[cou] = 0.0;
                        cou++;
                    }
                }
            }

            copyBuffer(latticeonevol, stagingBuffer,0, nVertsone * sizeof(float));
            vkUnmapMemory(device, stagingMemory);
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMemory, nullptr);
        }


 // ////////////////////position and normal ///////////////////////////////////
        {
            
            void *stagingBase;
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;
            VkDeviceSize stagingSz = maxmemvertsone * sizeof(float4);
            createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

            vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);
            
            float4 *posnorm = (float4 *)stagingBase;
            for (uint i=0;i<maxmemvertsone;i++)
            {
                posnorm[i].x=0.0f;
                posnorm[i].y=0.0f;
                posnorm[i].z=0.0f;
                posnorm[i].w=0.0f;
            }
            
            copyBuffer(vpos_one, stagingBuffer,0, maxmemvertsone * sizeof(float4));
        
            copyBuffer(vnorone, stagingBuffer,0, maxmemvertsone * sizeof(float4));
          
            vkUnmapMemory(device, stagingMemory);
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMemory, nullptr);
            
        }

    

        // ////////////////////position and normal ///////////////////////////////////
        {
            
            void *stagingBase;
            VkBuffer stagingBuffer;
            VkDeviceMemory stagingMemory;
            VkDeviceSize stagingSz = maxmemvertstwo * sizeof(float4);
            createBuffer(stagingSz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingMemory);

            vkMapMemory(device, stagingMemory, 0, stagingSz, 0, &stagingBase);
            
            float4 *posnorm = (float4 *)stagingBase;
            for (uint i=0;i<maxmemvertstwo;i++)
            {
                posnorm[i].x=0.0f;
                posnorm[i].y=0.0f;
                posnorm[i].z=0.0f;
                posnorm[i].w=0.0f;
            }
            
          
            copyBuffer(vpos_two, stagingBuffer,0, maxmemvertstwo * sizeof(float4));
          
            copyBuffer(vnortwo, stagingBuffer,0, maxmemvertstwo * sizeof(float4));
       
            vkUnmapMemory(device, stagingMemory);
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMemory, nullptr);
            
        }

        

        
        createExternalSemaphore(vkSignalSemaphore, getDefaultSemaphoreHandleType());
      
        createExternalSemaphore(vkWaitSemaphore, getDefaultSemaphoreHandleType());
      
        importCudaExternalSemaphore(cudaWaitSemaphore, vkSignalSemaphore, getDefaultSemaphoreHandleType());
      
        importCudaExternalSemaphore(cudaSignalSemaphore, vkWaitSemaphore, getDefaultSemaphoreHandleType());


    }
    
    void importCudaExternalMemory(void **cudaPtr, cudaExternalMemory_t& cudaMem, VkDeviceMemory& vkMem, VkDeviceSize size, VkExternalMemoryHandleTypeFlagBits handleType) 
    {
        cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};

        if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
            externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        }
        else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
            externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
        }
        else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
            externalMemoryHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        }
        else {
            throw std::runtime_error("Unknown handle type requested!");
        }

        externalMemoryHandleDesc.size = size;

        externalMemoryHandleDesc.handle.fd = (int)(uintptr_t)getMemHandle(vkMem, handleType);

        checkCudaErrors(cudaImportExternalMemory(&cudaMem, &externalMemoryHandleDesc));

        cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
        externalMemBufferDesc.offset = 0;
        externalMemBufferDesc.size = size;
        externalMemBufferDesc.flags = 0;
        checkCudaErrors(cudaExternalMemoryGetMappedBuffer(cudaPtr, cudaMem, &externalMemBufferDesc));
    }


    void importCudaExternalSemaphore(cudaExternalSemaphore_t& cudaSem, VkSemaphore& vkSem, VkExternalSemaphoreHandleTypeFlagBits handleType) 
    {
        cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};

        #ifdef _VK_TIMELINE_SEMAPHORE
        if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
            externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
        }
        else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
            externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
        }
        else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
            externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
        }

        #else

        if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
            externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
        }
        else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
            externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
        }
        else if (handleType & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
            externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
        }

        #endif /* _VK_TIMELINE_SEMAPHORE */

        else {
            throw std::runtime_error("Unknown handle type requested!");
        }

        #ifdef _WIN64
        externalSemaphoreHandleDesc.handle.win32.handle = (HANDLE)getSemaphoreHandle(vkSem, handleType);
        #else
        externalSemaphoreHandleDesc.handle.fd = (int)(uintptr_t)getSemaphoreHandle(vkSem, handleType);
        #endif

        externalSemaphoreHandleDesc.flags = 0;

        checkCudaErrors(cudaImportExternalSemaphore(&cudaSem, &externalSemaphoreHandleDesc));
    }

    void fillRenderingCommandBuffer_1(VkCommandBuffer& commandBuffer) 
    {
   
        VkBuffer vertexBuffers[] = { latticeonevol, xyzBufferone};
        VkDeviceSize offsets[] = { 0, 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBufferone, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, (uint32_t)(Nxu * Nyu * Nzu), 1, 0, 0, 0);
    
    }

    void fillRenderingCommandBuffer_2(VkCommandBuffer& commandBuffer) 
    {
   
        VkBuffer vertexBuffers[] = { latticetwovol,xyzBuffertwo };
        VkDeviceSize offsets[] = { 0, 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffertwo, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, (uint32_t)(NumX * NumY * NumZ), 1, 0, 0, 0);
    
    }

    void fillRenderingCommandBuffer_3(VkCommandBuffer& commandBuffer) 
    {
        VkBuffer vertexBuffers[] = { latticethreevol, xyzBufferthree };
        VkDeviceSize offsets[] = { 0, 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBufferthree, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, (uint32_t)(NumX2 * NumY2 * NumZ2), 1, 0, 0, 0);
    }


    void fillRenderingCommandBufferone(VkCommandBuffer& commandBuffer) 
    {
   
        VkBuffer vertexBuffers[] = {vpos_one, vnorone};
        VkDeviceSize offsets[] = { 0, 0 };
        vkCmdPushConstants(commandBuffer,pipelineLayout,VK_SHADER_STAGE_GEOMETRY_BIT,0,sizeof(LightPushConstants),&push_constant);
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
        vkCmdDraw(commandBuffer, (uint32_t)(totalVertsone), 1, 0, 0);
    }

    void fillRenderingCommandBuffertwo(VkCommandBuffer& commandBuffer) 
    {
   
        VkBuffer vertexBuffers[] = {vpos_two, vnortwo};
        VkDeviceSize offsets[] = { 0, 0 };
        vkCmdPushConstants(commandBuffer,pipelineLayout,VK_SHADER_STAGE_GEOMETRY_BIT,0,sizeof(LightPushConstants),&push_constant);
        vkCmdBindVertexBuffers(commandBuffer, 0, 2, vertexBuffers, offsets);
        vkCmdDraw(commandBuffer, (uint32_t)(totalVertstwo), 1, 0, 0);
    }


    VkDeviceSize getUniformSize() const 
    {
        return sizeof(UniformBufferObject);
    }


    void updateUniformBuffer(uint32_t imageIndex, bool shift) 
    {
        {
            
            float l1_r1 = dist1*0.7f;
            float b1_t1 = Nyu*0.7f;
            float n1_f1 = dist1*2.5f;

            float l2_r2 = dist2*0.7f;
            float b2_t2 = NumY*0.7f;
            float n2_f2 = dist2*2.5f;

            float l3_r3 = dist2*0.7f;
            float b3_t3 = NumY*0.7f;
            float n3_f3 = dist2*2.5f;

            angle1 += degreesToRadians(dist1*0.001f);
            float a_1 = (dist1*1.5f)*cos(angle1);
            float b_1 = (Nyu-1)/2.0f;
            float c_1 = (dist1*1.5f)*sin(angle1);

            angle2 += degreesToRadians(dist2*0.001f);
            float a_2 = (dist2*1.5f)*cos(angle2);
            float b_2 = (NumY-1)/2.0f;
            float c_2 = (dist2*1.5f)*sin(angle2);

            


            float a_3,b_3,c_3;
            if(!shift)
            {
                a_3 = (NumX-1)/2.0f;
                b_3 = (NumY-1)/2.0f;
                c_3 = (dist2*1.5f);
            }
            else
            {
                a_3 = (dist2*1.5f)*cos(angle3);
                b_3 = (NumY-1)/2.0f;
                c_3 = (dist2*1.5f)*sin(angle3);

                angle3 += degreesToRadians(dist2*0.001f);
            }

            mat4x4 view[5], proj[5];

            vec3 eye[5] = {{a_1, b_1, c_1},
                            {a_2, b_2, c_2},
                            {a_3, b_3, c_3},
                            {a_1, b_1, c_1},
                            {a_3, b_3, c_3}};
                     
            push_constant.eyes[0] = a_2*3;
            push_constant.eyes[1] = b_2;
            push_constant.eyes[2] = c_2*3;
            push_constant.eyes[3] = 1.0;

            vec3 center[5] = {
                              {float((Nxu-1)/2.0), float((Nyu-1)/2.0), float((Nzu-1)/2.0)},
                              {float((NumX-1)/2.0), float((NumY-1)/2.0), float((NumZ-1)/2.0)},
                              {float((NumX-1)/2.0), float((NumY-1)/2.0), float((NumZ-1)/2.0)},
                              {float((Nxu-1)/2.0), float((Nyu-1)/2.0), float((Nzu-1)/2.0)},
                              {float((NumX-1)/2.0), float((NumY-1)/2.0), float((NumZ-1)/2.0)}};
                          
            
            vec3 up[5] = {{ 0.0f, 1.0f, 0.0f },{ 0.0f, 1.0f, 0.0f },{ 0.0f, 1.0f, 0.0f },{ 0.0f, 1.0f, 0.0f },
            { 0.0f, 1.0f, 0.0f }};
          
            mat4x4_ortho(proj[0],-(l1_r1),(l1_r1),-(b1_t1),(b1_t1),-n1_f1,n1_f1);
            mat4x4_ortho(proj[1],-(l2_r2),(l2_r2),-(b2_t2),(b2_t2),-n2_f2,n2_f2);
            mat4x4_ortho(proj[2],-(l3_r3),(l3_r3),-(b3_t3),(b3_t3),-n3_f3,n3_f3);
            mat4x4_ortho(proj[3],-(l1_r1),(l1_r1),-(b1_t1),(b1_t1),-n1_f1,n1_f1);
            mat4x4_ortho(proj[4],-(l3_r3),(l3_r3),-(b3_t3),(b3_t3),-n3_f3,n3_f3);
          
            proj[0][1][1] *= -1.0f;        
            proj[1][1][1] *= -1.0f;
            proj[2][1][1] *= -1.0f;
            proj[3][1][1] *= -1.0f;
            proj[4][1][1] *= -1.0f;
          
            mat4x4_look_at(view[0], eye[0], center[0], up[0]);
            mat4x4_look_at(view[1], eye[1], center[1], up[1]);
            mat4x4_look_at(view[2], eye[2], center[2], up[2]);
            mat4x4_look_at(view[3], eye[3], center[3], up[3]);
            mat4x4_look_at(view[4], eye[4], center[4], up[4]);
     
            mat4x4_mul(ubo.modelViewProj[0], proj[0], view[0]);
            mat4x4_mul(ubo.modelViewProj[1], proj[1], view[1]);
            mat4x4_mul(ubo.modelViewProj[2], proj[2], view[2]);
            mat4x4_mul(ubo.modelViewProj[3], proj[3], view[3]);
            mat4x4_mul(ubo.modelViewProj[4], proj[4], view[4]);
      
        }
      
        void *data;
        vkMapMemory(device, uniformMemory[imageIndex], 0, getUniformSize(), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformMemory[imageIndex]);
    }

    std::vector<const char *> getRequiredExtensions() const 
    {
        std::vector<const char *> extensions;
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
        return extensions;
    }

    std::vector<const char *> getRequiredDeviceExtensions() const 
    {
        std::vector<const char *> extensions;
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
        extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME);

        return extensions;
    }


    void drawFrame(bool shift)
    {
        
        cudaExternalSemaphoreWaitParams waitParams = {};
        waitParams.flags = 0;
        waitParams.params.fence.value = 0;

        cudaExternalSemaphoreSignalParams signalParams = {};
        signalParams.flags = 0;
        signalParams.params.fence.value = 0;
       
        checkCudaErrors(cudaSignalExternalSemaphoresAsync(&cudaSignalSemaphore, &signalParams, 1));
        VulkanBaseApp::drawFrame(shift);
    
        checkCudaErrors(cudaWaitExternalSemaphoresAsync(&cudaWaitSemaphore, &waitParams, 1));
        
        
    }



    void initMC()
    {
       
        gridSizeone = make_uint3(Nxu, Nyu, Nzu);
        gridSizetwo = make_uint3(NumX2, NumY2, NumZ2);

        gridSizeMaskone = make_uint3(Nxu-1, Nyu-1, Nzu-1);
        gridSizeMasktwo = make_uint3(NumX2-1, NumY2-1, NumZ2-1);
      
        gridSizeShiftone = make_uint3(1,Nxu-1,(Nxu-1)*(Nyu-1));
        gridSizeShifttwo = make_uint3(1,NumX2-1,(NumX2-1)*(NumY2-1));
    
        numVoxelsone = gridSizeMaskone.x*gridSizeMaskone.y*gridSizeMaskone.z;
        numVoxelstwo = gridSizeMasktwo.x*gridSizeMasktwo.y*gridSizeMasktwo.z;
  
        voxelSizeone = make_float3(1.0,1.0,1.0);
        voxelSizetwo = make_float3(dx2,dy2,dz2);
  
        gridcenterone = make_float3(0.0,0.0,0.0);
        gridcentertwo = make_float3(0.0,0.0,0.0);

        mcalgo.allocateTextures_s(&d_triTable, &d_numVertsTable);
  
        unsigned int memSizeone = sizeof(uint) * numVoxelsone ;
        unsigned int memSizetwo = sizeof(uint) * numVoxelstwo ;
     
        checkCudaErrors(cudaMalloc((void **) &d_voxelVertsone,            memSizeone));
        checkCudaErrors(cudaMalloc((void **) &d_voxelVertsScanone,        memSizeone));
        checkCudaErrors(cudaMalloc((void **) &d_voxelOccupiedone,         memSizeone));
        checkCudaErrors(cudaMalloc((void **) &d_voxelOccupiedScanone,     memSizeone));
        checkCudaErrors(cudaMalloc((void **) &d_compVoxelArrayone,   memSizeone));

        checkCudaErrors(cudaMalloc((void **) &d_voxelVertstwo,            memSizetwo));
        checkCudaErrors(cudaMalloc((void **) &d_voxelVertsScantwo,        memSizetwo));
        checkCudaErrors(cudaMalloc((void **) &d_voxelOccupiedtwo,         memSizetwo));
        checkCudaErrors(cudaMalloc((void **) &d_voxelOccupiedScantwo,     memSizetwo));
        checkCudaErrors(cudaMalloc((void **) &d_compVoxelArraytwo,   memSizetwo));

    
    }

    void computeIsosurface(float* vol, float4* pos , float4* norm, float isoValue,
    uint numVoxels, uint *d_voxelVerts,uint *d_voxelVertsScan, uint *d_voxelOccupied,uint *d_voxelOccupiedScan,
    uint3 gridSize,uint3 gridSizeShift,uint3 gridSizeMask, float3 voxelSize, float3 gridcenter,
    uint *activeVoxels, uint *totalVerts, uint *d_compVoxelArray, uint maxVerts, float* vol_one)
    {
        
        dim3 grid(ceil(numVoxels/float(1024)), 1, 1);
        dim3 threads(1024,1,1);
      
        if (grid.x > 65535)
        {
            grid.y = grid.x / 32768;
            grid.x = 32768;
        }
    
        mcalgo.classifyVoxel_lattice(grid,threads,
                            d_voxelVerts, d_voxelOccupied, vol,
                            gridSize, gridSizeShift, gridSizeMask,
                            numVoxels, voxelSize, isoValue);
 
        ////// Numbering active voxels ///////
        
        mcalgo.ThrustScanWrapper_lattice(d_voxelOccupiedScan, d_voxelOccupied, numVoxels);
    
        {
            uint lastElement, lastScanElement;
            checkCudaErrors(cudaMemcpy((void *) &lastElement,
                                    (void *)(d_voxelOccupied + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy((void *) &lastScanElement,
                                    (void *)(d_voxelOccupiedScan + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            *activeVoxels = lastElement + lastScanElement;
            //printf("activevoxel %d lastElement %u lastScanElement %u \n",*activeVoxels,lastElement,lastScanElement);
        }

        if (*activeVoxels == 0)
        {
            *totalVerts = 0;
            return;
        }
    
        dim3 gids(ceil((numVoxels)/float(1024)), 1, 1);
        dim3 tids(1024,1,1);
    
    
        mcalgo.compactVoxels_lattice(gids, tids, d_compVoxelArray, d_voxelOccupied, d_voxelOccupiedScan, numVoxels);
        getLastCudaError("compactVoxels failed");

        /////////Finding totalverts ////////////

        mcalgo.ThrustScanWrapper_lattice(d_voxelVertsScan, d_voxelVerts, numVoxels);

        
    
        {
            uint totalverts_1;
            uint lastElement_1, lastScanElement_1;
            
            checkCudaErrors(cudaMemcpy((void *) &lastElement_1,
                                    (void *)(d_voxelVerts + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy((void *) &lastScanElement_1,
                                    (void *)(d_voxelVertsScan + (numVoxels)-1),
                                    sizeof(uint), cudaMemcpyDeviceToHost));

            totalverts_1 = lastElement_1 + lastScanElement_1;
            
            *totalVerts = totalverts_1;
            
        }

        // printf("totalverts %d \n",*totalVerts);

       
        dim3 grid2((int) ceil(*activeVoxels / (float)NTHREADS), 1, 1);
       
        dim3 tids2(NTHREADS,1,1);
       
        mcalgo.generateTriangles_lattice(grid2, tids2, pos, norm,
                                d_compVoxelArray,
                                d_voxelVertsScan, vol,
                                gridSize, gridSizeShift, gridSizeMask,
                                voxelSize,gridcenter, isoValue, *activeVoxels,
                                maxVerts,*totalVerts,vol_one);

   
        
        
        
    }

    
    int lattice_init(){

        fftlattice.create_lattice(d_volumeone,Nxu,Nyu,Nzu,sizeone);
        lattice.normalise_buffer(d_volumeone,d_volumeone,Nxu*Nyu*Nzu);
       
        checkCudaErrors(cufftPlan3d(&planr2c, Nxu, Nyu, Nzu, CUFFT_R2C));
        
        checkCudaErrors(cufftPlan3d(&planc2r, Nxu, Nyu, Nzu, CUFFT_C2R));
        checkCudaErrors(cudaMalloc((void **)&fft_data, sizeof(float2) * (Nxu * Nyu * ((Nzu/2)+1))));
        checkCudaErrors(cudaMalloc((void **)&fft_data_compute, sizeof(float2) * (Nxu * Nyu * ((Nzu/2)+1))));
        checkCudaErrors(cudaMalloc((void **)&fft_data_compute_fill, sizeof(float2) * (Nxu * Nyu * Nzu)));
        checkCudaErrors(cudaMalloc((void **)&fft_gratings, sizeof(float) * (Nxu * Nyu * Nzu)));
        checkCudaErrors(cudaMalloc((void **)&lattice_data, sizeof(float2) * (indi_range*indi_range*indi_range)));

        checkCudaErrors(cudaMalloc((void **)&d_volumeone_one, sizeof(float) * (Nxu*Nyu*Nzu)));
        checkCudaErrors(cudaMalloc((void **)&d_volumethree_one, sizeof(float) * (NumX2*NumY2*NumZ2)));

        cudaMemset(fft_data,0,sizeof(float2) * (Nxu * Nyu * ((Nzu/2)+1)));
        cudaMemset(fft_data_compute,0,sizeof(float2) * (Nxu * Nyu * ((Nzu/2)+1)));
        cudaMemset(fft_data_compute_fill,0,sizeof(float2) * (Nxu * Nyu * Nzu));
        cudaMemset(fft_gratings,0,sizeof(float) * (Nxu * Nyu * Nzu));
        cudaMemset(lattice_data,0,sizeof(float2) * (indi_range*indi_range*indi_range));

        cudaMemset(d_volumeone_one,0,sizeof(float) * (Nxu*Nyu*Nzu));
        cudaMemset(d_volumethree_one,0,sizeof(float) * (NumX2*NumY2*NumZ2));

        checkCudaErrors(cudaMemcpy2D(fft_data,(Nxu+1)*sizeof(float), d_volumeone,(Nxu*sizeof(float)), (Nxu*sizeof(float)),Nyu*Nzu, cudaMemcpyDeviceToDevice));
      
        checkCudaErrors(cudaMalloc((void **)&d_phi, sizeof(float) * (NumX*NumY*NumZ)));
       
        checkCudaErrors(cudaMalloc((void **)&d_ga, sizeof(float2) * (NumX2*NumY2*NumZ2)));
        checkCudaErrors(cudaMalloc((void **)&d_svl, sizeof(float) * (NumX2*NumY2*NumZ2)));
      
        checkCudaErrors(cudaMalloc((void **)&d_theta, sizeof(float) * (NumX*NumY*NumZ)));
        checkCudaErrors(cudaMalloc((void **)&d_period, sizeof(float) * (NumX*NumY*NumZ)));

        cudaMemset(d_phi,0,sizeof(float) * size);
     

        cudaMemset(d_svl,0,sizeof(float) * size2);


        float mn_x = 0;
        float mn_y = 0;
        float mn_z = 0;

        if((latticetype_one == 'r'))
        {
            mn_x = (NumX/2.0);
            mn_y = (NumY/2.0);
            mn_z = (NumZ/2.0);
        }
       
        
        lattice.angle_data(d_theta,NumX,NumY,NumZ,dx,dy,dz,mn_x,mn_y,mn_z,'z');
        lattice.period_data(d_period,NumX,NumY,NumZ,dx,dy,dz,mn_x,mn_y,mn_z,'z');
        lattice.normalise_bufferthree(d_period,d_period,size,NumX/10,NumX/4);

        size_t d_width = NumX, d_height = NumY, d_depth = NumZ;
        extend = make_cudaExtent(d_width*sizeof(float),d_height, d_depth);
        
        cudaMalloc3D(&devPitchedPtr, extend);
        getLastCudaError("cudaMalloc3D failed");
        tPitch = devPitchedPtr.pitch;

        slicepitch = tPitch*d_height;
    
        cudaMemcpy3DParms params ={0};
        params.srcPtr = make_cudaPitchedPtr(d_phi,d_width*sizeof(float),d_width,d_height);
        params.dstPtr = devPitchedPtr;
        params.extent = extend;
        params.kind = cudaMemcpyDeviceToDevice;

        cudaMemcpy3D( &params );
        getLastCudaError("cudaMemcpy3D failed ");

        lattice.setupTexture(NumX,NumY,NumZ);

        printf("Initialisation Completed Successfully \n\n");
      
        return 0;
    }

    int unit_lattice()
    {
        fftlattice.fft_func(fft_data);
        checkCudaErrors(cudaMemcpy(fft_data_compute, fft_data, (((Nxu/2)+1) * Nyu * Nzu) * sizeof(float2), cudaMemcpyDeviceToDevice));
        fftlattice.fft_scalar(fft_data_compute,sizeone,(((Nxu/2)+1) * Nyu * Nzu));
        fftlattice.fft_fill(fft_data_compute,fft_data_compute_fill,Nxu,Nyu,Nzu);
    
        fftlattice.ifft_func(fft_data);
        
        // ////////////////////////////////
        float2 *h_dat;
        
        h_dat = (float2 *)malloc((Nxu*Nyu*Nzu) * sizeof(float2));
        checkCudaErrors(cudaMemcpy(h_dat, fft_data_compute_fill, (Nxu * Nyu * Nzu) * sizeof(float2), cudaMemcpyDeviceToHost));

        float2 *h_lattice_dat;
        h_lattice_dat = (float2 *)malloc((indi_range*indi_range*indi_range) * sizeof(float2));

       
        int l_count = 0;
        int midd1 = floor((Nxu*Nyu*Nzu)/2.0);
        
        int midd2 = floor((indi_range*indi_range*indi_range)/2.0);
        for(int k = -range_st;k <(range_st+1);k++)
        {
                for(int j = -range_st;j <(range_st+1);j++)
                {
                
                        for(int i = -range_st;i <(range_st+1);i++)
                        {
                         

                            if((i == 0) && (j == 0) && (k == 0))
                            {
                                h_lattice_dat[midd2] = h_dat[0];
                                
                                
                              
                                l_count++;
                                continue;
                            }
                           
                            int u,s,t;
                            u = floor(Nxu/2.0) + i;
                            s = floor(Nyu/2.0) + j;
                            t = floor(Nzu/2.0) + k;

                            if( u == 15)
                            {
                                u = 0;
                            }
                            else if( u > 15)
                            {
                                u -= 15;
                            }
                            else
                            {
                                u += 16;
                            }

                            if( s == 15)
                            {
                                s = 0;
                            }
                            else if( s > 15)
                            {
                                s -= 15;
                            }
                            else
                            {
                                s += 16;
                            }

                            if( t == 15)
                            {
                                t = 0;
                            }
                            else if( t > 15)
                            {
                                t -= 15;
                            }
                            else
                            {
                                t += 16;
                            }
                
                            int indx2 = u + s * 31 + t * (31*31);

                            h_lattice_dat[l_count] = h_dat[indx2];

                            l_count++;
                        

                        }
                    
                }

        }

        checkCudaErrors(cudaMemcpy2D(fft_gratings,Nxu * sizeof(float), fft_data,(((Nzu/2)+1)*2)*sizeof(float), (Nxu*sizeof(float)),Nyu*Nzu, cudaMemcpyDeviceToDevice));

        lattice.normalise_buffer(fft_gratings,fft_gratings,Nxu*Nyu*Nzu);
     
        lattice.normalise_bufferfour(fft_gratings,d_volumeone_one,Nxu*Nyu*Nzu,Nxu,Nyu,Nzu,isoValue);
        
        computeIsosurface(d_volumeone_one,d_posone,d_normalone,isoValue,
        numVoxelsone,d_voxelVertsone,d_voxelVertsScanone, d_voxelOccupiedone,d_voxelOccupiedScanone,
        gridSizeone,gridSizeShiftone,gridSizeMaskone, voxelSizeone,gridcenterone,
        &activeVoxelsone, &totalVertsone, d_compVoxelArrayone,maxmemvertsone,fft_gratings);

        checkCudaErrors(cudaMemcpy(lattice_data,h_lattice_dat,sizeof(float2)*(indi_range*indi_range*indi_range),cudaMemcpyHostToDevice));
        
        checkCudaErrors(cudaFree(fft_data));
        checkCudaErrors(cudaFree(fft_data_compute));
        checkCudaErrors(cudaFree(fft_data_compute_fill));
        free(h_lattice_dat);

        cudaExternalSemaphoreWaitParams waitParams = {};
        waitParams.flags = 0;
        waitParams.params.fence.value = 0;

        cudaExternalSemaphoreSignalParams signalParams = {};
        signalParams.flags = 0;
        signalParams.params.fence.value = 0;
    
        checkCudaErrors(cudaSignalExternalSemaphoresAsync(&cudaSignalSemaphore, &signalParams, 1));
        VulkanBaseApp::updatecommandBuffers();
        VulkanBaseApp::drawFrame(shift);
        checkCudaErrors(cudaWaitExternalSemaphoresAsync(&cudaWaitSemaphore, &waitParams, 1));
        
        return 0;

    }

    
    ////////////////////////////////////////////////////////////////////////////////
    int spatial_lattice_run()
    {

      
        if (indi_range%2 == 0)
        {
            printf("Truncated Fft Matrix should have odd range. Exiting! \n");
            return -1;
        }
        

        cudaExternalSemaphoreWaitParams waitParams = {};
        waitParams.flags = 0;
        waitParams.params.fence.value = 0;

        cudaExternalSemaphoreSignalParams signalParams = {};
        signalParams.flags = 0;
        signalParams.params.fence.value = 0;

        int mycount = 0;
        int p0 = floor(Nxu/2.0);
        int q0 = floor(Nyu/2.0);
        int r0 = floor(Nzu/2.0);

        for(int k = -range_st;k <(range_st+1);k++)
        {
                for(int j = -range_st;j <(range_st+1);j++)
                {
                
                        for(int i = -range_st;i <(range_st+1);i++)
                        {
                           
                            lattice.finding_phi(d_phi,d_period,NumX,NumY,NumZ,i,j,k,dx,dy,dz,latticetype_one,latticetype_two);
                        
                            lattice.GPUCG_lattice(d_phi,iter,1,EndRes, FinalIter, FinalRes);

                            lattice.normalise_buffer(d_phi,d_volumetwo,NumX*NumY*NumZ);
                
                            lattice.copytotexture(d_phi,devPitchedPtr,NumX,NumY,NumZ);

                            lattice.updateTexture(devPitchedPtr);

                            lattice.grating(d_ga,NumX2,NumY2,NumZ2,dx2,dy2,dz2);

                            lattice.svl(d_svl,d_ga,NumX2,NumY2,NumZ2,mycount,lattice_data);
        
                            lattice.normalise_bufferfour(d_svl,d_volumethree_one,NumX2*NumY2*NumZ2,NumX2,NumY2,NumZ2,isoValue);
                            
                            lattice.normalise_buffer(d_svl,d_volumethree,NumX2*NumY2*NumZ2);
    
                            computeIsosurface(d_volumethree_one,d_postwo,d_normaltwo,isoValue,
                            numVoxelstwo,d_voxelVertstwo,d_voxelVertsScantwo, d_voxelOccupiedtwo,d_voxelOccupiedScantwo,
                            gridSizetwo,gridSizeShifttwo,gridSizeMasktwo, voxelSizetwo,gridcentertwo,
                            &activeVoxelstwo, &totalVertstwo, d_compVoxelArraytwo,maxmemvertstwo,
                            d_volumethree);
                        
                            checkCudaErrors(cudaSignalExternalSemaphoresAsync(&cudaSignalSemaphore, &signalParams, 1));
                            VulkanBaseApp::updatecommandBuffers();
                            VulkanBaseApp::drawFrame(shift);
                            checkCudaErrors(cudaWaitExternalSemaphoresAsync(&cudaWaitSemaphore, &waitParams, 1));
                            
                            mycount++;
                          
                        }
                }

        }
     
        if(ouput_data)
        {
            const char *filenameone = "Unit_Lattice.obj";
            output_file.file_write(d_posone,totalVertsone,filenameone);

            const char *filenametwo = "Spatial_Lattice.obj";
            output_file.file_write(d_postwo,totalVertstwo,filenametwo);
        }

        return 0;
    }

    int compute_n_visualise()
    {
        unit_lattice();
        spatial_lattice_run();
        shift = true;
        VulkanBaseApp::mainLoop(shift);

        return 0;
    }


    void cleanup()
    {
            
        mcalgo.destroyAllTextureObjects();
        lattice.GPUCleanUp();
        checkCudaErrors(cudaFree(d_edgeTable));
        checkCudaErrors(cudaFree(d_triTable));
        checkCudaErrors(cudaFree(d_numVertsTable));

        checkCudaErrors(cudaFree(d_voxelVertsone));
        checkCudaErrors(cudaFree(d_voxelVertstwo));
    
        checkCudaErrors(cudaFree(d_voxelVertsScanone));
        checkCudaErrors(cudaFree(d_voxelVertsScantwo));
    
        checkCudaErrors(cudaFree(d_voxelOccupiedone));
        checkCudaErrors(cudaFree(d_voxelOccupiedtwo));

        checkCudaErrors(cudaFree(d_voxelOccupiedScanone));
        checkCudaErrors(cudaFree(d_voxelOccupiedScantwo));

        checkCudaErrors(cudaFree(d_compVoxelArrayone));
        checkCudaErrors(cudaFree(d_compVoxelArraytwo));

        checkCudaErrors(cudaFree(devPitchedPtr.ptr));

        checkCudaErrors(cudaFree(d_phi));
    
        checkCudaErrors(cudaFree(d_ga));
        checkCudaErrors(cudaFree(d_svl));
    
        checkCudaErrors(cudaFree(d_theta));
        checkCudaErrors(cudaFree(d_period));

        checkCudaErrors(cudaFree(fft_gratings));
        checkCudaErrors(cudaFree(lattice_data));
    
        if (d_volumeone)
        {
            checkCudaErrors(cudaFree(d_volumeone));
        }

        if (d_volumetwo)
        {
            checkCudaErrors(cudaFree(d_volumetwo));
        }

        if (d_volumethree)
        {
            checkCudaErrors(cudaFree(d_volumethree));
        }

        if (d_volumeone_one)
        {
            checkCudaErrors(cudaFree(d_volumeone_one));
        }

        if (d_volumethree_one)
        {
            checkCudaErrors(cudaFree(d_volumethree_one));
        }

        if (d_posone)
        {
            checkCudaErrors(cudaFree(d_posone));
        }

        if (d_postwo)
        {
            checkCudaErrors(cudaFree(d_postwo));
        }

        if (d_normalone)
        {
            checkCudaErrors(cudaFree(d_normalone));
        }

        if (d_normaltwo)
        {
            checkCudaErrors(cudaFree(d_normaltwo));
        }
   
    };

};




int main(int argc, char** argv)
{
    int grid_value;
    float grid_spacing;
    char a;
    char b;
    bool data_out;

    grid_value = atoi(argv[1]);

    if((grid_value < 16) || (grid_value > 150))
    {
        printf("Grid Dimension should be in the interval 16 and 150 \n");
        return 0;

    }

    grid_spacing = atof(argv[2]);

    if(grid_spacing != 1.0)
    {
        printf("Grid Spacing Value should be 1.0 \n");
        return 0;
    }

    a = *argv[3];

    if((a != 'b') && (a != 'r') && (a != 'n'))
    {
        printf(" Unknown lattice type \n");
        return 0;
    }


    b = *argv[4];

    if((b != 'u') && (b != 'v') )
    {
        printf(" lattice type neither uniform nor variable \n");
        return 0;
    }

    std::string output_data = argv[5];

    if(output_data == "true")
    {
        data_out = true;
    }
    else if(output_data == "false")
    {
        data_out = false;
    }
    else
    {
        printf(" Unknown bool type. Niether True nor False \n");
        return 0;
    }

    Mutlitopo app(grid_value, grid_value, grid_value,grid_spacing,grid_spacing,grid_spacing,a,b,data_out);
    app.init();
    app.initMC();
    app.lattice_init();
    app.compute_n_visualise();
    app.cleanup();
    printf("Done !\n");
    return 0;


}












