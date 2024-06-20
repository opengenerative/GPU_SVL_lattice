/*Reference :- https://github.com/NVIDIA/cuda-samples/blob/master/Samples/5_Domain_Specific/simpleVulkan/VulkanBaseApp.cpp*/

#include <unistd.h>
#include <typeinfo>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <set>
#include <string.h>
#include <limits>

#include "VulkanApp.h"
#define GLFW_INCLUDE_VULKAN
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <GLFW/glfw3.h>

#ifndef countof
#define countof(x) (sizeof(x) / sizeof(*(x)))
#endif


static const char *validationLayers[] = { "VK_LAYER_KHRONOS_validation" };
static const size_t MAX_FRAMES_IN_FLIGHT = 5;


struct 
{
    VkPipeline graphicsPipeline_1;
    VkPipeline graphicsPipelineone;
    VkPipeline graphicsPipeline_2;
    VkPipeline graphicsPipelinetwo;
    VkPipeline graphicsPipeline_3;

} pipelines;



void VulkanBaseApp::resizeCallback(GLFWwindow *window, int width, int height)
{
    VulkanBaseApp *app = reinterpret_cast<VulkanBaseApp *>(glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
{
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

    return VK_FALSE;
}

VulkanBaseApp::VulkanBaseApp(const std::string& appName, bool enableValidation) :
    appName(appName),
    enableValidation(enableValidation),
    vpcount(5),
    instance(VK_NULL_HANDLE),
    window(nullptr),
    debugMessenger(VK_NULL_HANDLE),
    surface(VK_NULL_HANDLE),
    physicalDevice(VK_NULL_HANDLE),
    device(VK_NULL_HANDLE),
    graphicsQueue(VK_NULL_HANDLE),
    presentQueue(VK_NULL_HANDLE),
    swapChain(VK_NULL_HANDLE),
    vkDeviceUUID(),
    swapChainImages(),
    swapChainFormat(),
    swapChainExtent(),
    swapChainImageViews(),
    shaderFiles_1(),
    shaderFilesone(),
    shaderFiles_2(),
    shaderFilestwo(),
    shaderFiles_3(),
    renderPass(),
    pipelineLayout(VK_NULL_HANDLE),
    swapChainFramebuffers(),
    viewports(),
    scissors(),
    commandPool(VK_NULL_HANDLE),
    commandBuffers(),
    imageAvailableSemaphores(),
    renderFinishedSemaphores(),
    inFlightFences(),
    uniformBuffers(),
    uniformMemory(),
    descriptorSetLayout(VK_NULL_HANDLE),
    descriptorPool(VK_NULL_HANDLE),
    descriptorSets(),
    depthImage(VK_NULL_HANDLE),
    depthImageMemory(VK_NULL_HANDLE),
    depthImageView(VK_NULL_HANDLE),
    currentFrame(0),
    framebufferResized(false)
    
{
}

VkExternalSemaphoreHandleTypeFlagBits VulkanBaseApp::getDefaultSemaphoreHandleType()
{

    return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

}

VkExternalMemoryHandleTypeFlagBits VulkanBaseApp::getDefaultMemHandleType()
{

    return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

}

VulkanBaseApp::~VulkanBaseApp()
{
  
    cleanupSwapChain();

    if (descriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    }

    for (size_t i = 0; i < renderFinishedSemaphores.size(); i++) {
       
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }
    if (commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, nullptr);
    }

    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
    }

    if (enableValidation) {
        PFN_vkDestroyDebugUtilsMessengerEXT func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func != nullptr) {
            func(instance, debugMessenger, nullptr);
        }
    }

    if (surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
    }

    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
    }

    if (window) {
        glfwDestroyWindow(window);
    }
  

    glfwTerminate();
}

void VulkanBaseApp::init()
{
  
    initWindow();
    initVulkan();
   
}

VkCommandBuffer VulkanBaseApp::beginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void VulkanBaseApp::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(graphicsQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

void VulkanBaseApp::initWindow()
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window = glfwCreateWindow(2000, 2000, appName.c_str(), nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, resizeCallback);
    glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, GL_FALSE);
}


std::vector<const char *> VulkanBaseApp::getRequiredExtensions() const
{
    return std::vector<const char *>();
}

std::vector<const char *> VulkanBaseApp::getRequiredDeviceExtensions() const
{
    return std::vector<const char *>();
}

void VulkanBaseApp::initVulkan()
{
    createInstance();
    createSurface();
    createDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createCommandPool();
    createDepthResources();
    createFramebuffers();
    initVulkanApp();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
    
}


static VkFormat findSupportedFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
{
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);
        if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }
    throw std::runtime_error("Failed to find supported format!");
}

static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return ~0;
}

static bool supportsValidationLayers()
{
    std::vector<VkLayerProperties> availableLayers;
    uint32_t layerCount;

    vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
    availableLayers.resize(layerCount);
    vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

    for (const char * layerName : validationLayers) {
        bool layerFound = false;

        for (const auto & layerProperties : availableLayers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layerFound = true;
                break;
            }
        }

        if (!layerFound) {
            return false;
        }
    }

    return true;
}



void VulkanBaseApp::createInstance()
{
    if (enableValidation && !supportsValidationLayers()) {
        throw std::runtime_error("Validation requested, but not supported!");
    }

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = appName.c_str();
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    

    std::vector<const char *> exts = getRequiredExtensions();

    {
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        
        exts.insert(exts.begin(), glfwExtensions, glfwExtensions + glfwExtensionCount);

        
        if (enableValidation) {
            exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            
        }


    }

    createInfo.enabledExtensionCount = static_cast<uint32_t>(exts.size());
    createInfo.ppEnabledExtensionNames = exts.data();

 

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo = {};
    if (enableValidation) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(countof(validationLayers));
        createInfo.ppEnabledLayerNames = validationLayers;

        debugCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugCreateInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugCreateInfo.pfnUserCallback = debugCallback;
        
        createInfo.pNext = &debugCreateInfo;
    }
    else {
        createInfo.enabledLayerCount = 0;
        createInfo.pNext = nullptr;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance!");
    }
    
    if (enableValidation) {
        PFN_vkCreateDebugUtilsMessengerEXT func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
        if (func == nullptr || func(instance, &debugCreateInfo, nullptr, &debugMessenger) != VK_SUCCESS) {
            throw std::runtime_error("Failed to set up debug messenger!");
        }
    }
}

void VulkanBaseApp::createSurface()
{
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }
}

static bool findGraphicsQueueIndicies(VkPhysicalDevice device, VkSurfaceKHR surface, uint32_t& graphicsFamily, uint32_t& presentFamily)
{
    uint32_t queueFamilyCount = 0;

    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

    graphicsFamily = presentFamily = ~0;

    for (uint32_t i = 0; i < queueFamilyCount; i++) {

        if (queueFamilies[i].queueCount > 0) {
            if (graphicsFamily == ~0 && queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                graphicsFamily = i;
            }
            uint32_t presentSupport = 0;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
            if (presentFamily == ~0 && presentSupport) {
                presentFamily = i;
            }
            if (presentFamily != ~0 && graphicsFamily != ~0) {
                break;
            }
        }
    }

    return graphicsFamily != ~0 && presentFamily != ~0;
}

static bool hasAllExtensions(VkPhysicalDevice device, const std::vector<const char *>& deviceExtensions)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

    std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto & extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }

    return requiredExtensions.empty();
}

static void getSwapChainProperties(VkPhysicalDevice device, VkSurfaceKHR surface, VkSurfaceCapabilitiesKHR& capabilities, std::vector<VkSurfaceFormatKHR>& formats, std::vector<VkPresentModeKHR>& presentModes)
{
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &capabilities);
    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);
    if (formatCount != 0) {
        formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, formats.data());
    }
    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);
    if (presentModeCount != 0) {
        presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, presentModes.data());
    }
}

bool VulkanBaseApp::isSuitableDevice(VkPhysicalDevice dev) const
{
    uint32_t graphicsQueueIndex, presentQueueIndex;
    std::vector<const char *> deviceExtensions = getRequiredDeviceExtensions();
    VkSurfaceCapabilitiesKHR caps;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    getSwapChainProperties(dev, surface, caps, formats, presentModes);
    return hasAllExtensions(dev, deviceExtensions)
           && !formats.empty() && !presentModes.empty()
           && findGraphicsQueueIndicies(dev, surface, graphicsQueueIndex, presentQueueIndex);
}

void VulkanBaseApp::createDevice()
{
    {
     
        uint32_t deviceCount = 0;

        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) {
            throw std::runtime_error("Failed to find Vulkan capable GPUs!");
        }
  
        //creating a vector to store physical devices data
        std::vector<VkPhysicalDevice> phyDevs(deviceCount);
   
        // filling vector with physVkPhysicalDevice as elements
        VkResult str  = vkEnumeratePhysicalDevices(instance, &deviceCount, phyDevs.data());
      
        for (const auto& device : phyDevs)
        {
        auto props = VkPhysicalDeviceProperties{};
        vkGetPhysicalDeviceProperties(device, &props);
        
        // Determine the type of the physical device
        if (props.deviceType == VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            
            physicalDevice = device;
            printf("Selected NVIDIA GPU \n\n");
            break;
        }

        else if (props.deviceType == VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
        {
            
            printf(" \nNot selecting Integrated GPU \n\n");
        }
        else if (props.deviceType == VkPhysicalDeviceType::VK_PHYSICAL_DEVICE_TYPE_CPU)
        {
         
            printf("Not selecting VK_PHYSICAL_DEVICE_TYPE_CPU \n\n");
        }

        else
        {
            printf(" Device not in Vulkan list \n");
        }

       
        }
       

        if (physicalDevice == VK_NULL_HANDLE){

            throw std::runtime_error(" Failed to set Physical device");
        }

    }

    uint32_t graphicsQueueIndex, presentQueueIndex;
    findGraphicsQueueIndicies(physicalDevice, surface, graphicsQueueIndex, presentQueueIndex);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueFamilyIndices = { graphicsQueueIndex, presentQueueIndex };

    float queuePriority = 1.0f;

    for (uint32_t queueFamily : uniqueFamilyIndices) {
        VkDeviceQueueCreateInfo queueCreateInfo = {};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = graphicsQueueIndex;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures = {};
    deviceFeatures.fillModeNonSolid = true;
    //myline
    deviceFeatures.geometryShader = true;
    deviceFeatures.multiViewport = true;
    deviceFeatures.wideLines = true;
    deviceFeatures.shaderTessellationAndGeometryPointSize = true;

    VkDeviceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());

    createInfo.pEnabledFeatures = &deviceFeatures;

    std::vector<const char *> deviceExtensions = getRequiredDeviceExtensions();
    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (enableValidation) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(countof(validationLayers));
        createInfo.ppEnabledLayerNames = validationLayers;
    }
    else {
        createInfo.enabledLayerCount = 0;
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("failed to create logical device!");
    }

    vkGetDeviceQueue(device, graphicsQueueIndex, 0, &graphicsQueue);
    vkGetDeviceQueue(device, presentQueueIndex, 0, &presentQueue);

    VkPhysicalDeviceIDProperties vkPhysicalDeviceIDProperties = {};
    vkPhysicalDeviceIDProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    vkPhysicalDeviceIDProperties.pNext = NULL;
    

    VkPhysicalDeviceProperties2 vkPhysicalDeviceProperties2 = {};
    vkPhysicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    vkPhysicalDeviceProperties2.pNext = &vkPhysicalDeviceIDProperties;

    PFN_vkGetPhysicalDeviceProperties2 fpGetPhysicalDeviceProperties2;
    fpGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2");
    if (fpGetPhysicalDeviceProperties2 == NULL) {
        throw std::runtime_error("Vulkan: Proc address for \"vkGetPhysicalDeviceProperties2KHR\" not found.\n");
    }

    fpGetPhysicalDeviceProperties2(physicalDevice, &vkPhysicalDeviceProperties2);

    memcpy(vkDeviceUUID, vkPhysicalDeviceIDProperties.deviceUUID,  VK_UUID_SIZE);

    
}

static VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
{
    if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED) {
        return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
    }

    for (const auto & availableFormat : availableFormats) {
        if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return availableFormat;
        }
    }

    return availableFormats[0];
}

static VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
{
    VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

    for (const auto & availablePresentMode : availablePresentModes) {
        if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
            return availablePresentMode;
        }
        else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR) {
            bestMode = availablePresentMode;
        }
    }

    return bestMode;
}

static VkExtent2D chooseSwapExtent(GLFWwindow *window, const VkSurfaceCapabilitiesKHR& capabilities)
{
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }
    else {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        VkExtent2D actualExtent = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

        actualExtent.width = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actualExtent.width));
        actualExtent.height = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actualExtent.height));

        return actualExtent;
    }
}

void VulkanBaseApp::createSwapChain()
{
    VkSurfaceCapabilitiesKHR capabilities;
    VkSurfaceFormatKHR format;
    VkPresentModeKHR presentMode;
    VkExtent2D extent;
    uint32_t imageCount;

    {
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;

        getSwapChainProperties(physicalDevice, surface, capabilities, formats, presentModes);
        format = chooseSwapSurfaceFormat(formats);
        presentMode = chooseSwapPresentMode(presentModes);
        extent = chooseSwapExtent(window, capabilities);
        imageCount = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
            imageCount = capabilities.maxImageCount;
        }
    }

    VkSwapchainCreateInfoKHR createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;

    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = format.format;
    createInfo.imageColorSpace = format.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    uint32_t queueFamilyIndices[2];
    findGraphicsQueueIndicies(physicalDevice, surface, queueFamilyIndices[0], queueFamilyIndices[1]);

    if (queueFamilyIndices[0] != queueFamilyIndices[1]) {
        createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createInfo.queueFamilyIndexCount = countof(queueFamilyIndices);
        createInfo.pQueueFamilyIndices = queueFamilyIndices;
    }
    else {
        createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;

    createInfo.oldSwapchain = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
        throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
    swapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

    swapChainFormat = format.format;
    swapChainExtent = extent;
}

static VkImageView createImageView(VkDevice dev, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags)
{
    VkImageView imageView;
    VkImageViewCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = image;
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = format;
    createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    createInfo.subresourceRange.aspectMask = aspectFlags;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;
    if (vkCreateImageView(dev, &createInfo, nullptr, &imageView) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create image views!");
    }

    return imageView;
}

static void createImage(VkPhysicalDevice physicalDevice, VkDevice device, uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
{
    VkImageCreateInfo imageInfo = {};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = tiling;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS) {
        throw std::runtime_error("failed to create image!");
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(device, image, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate image memory!");
    }

    vkBindImageMemory(device, image, imageMemory, 0);
}

void VulkanBaseApp::createImageViews()
{
    swapChainImageViews.resize(swapChainImages.size());

    for (uint32_t i = 0; i < swapChainImages.size(); i++) {
        swapChainImageViews[i] = createImageView(device, swapChainImages[i], swapChainFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }
}

void VulkanBaseApp::createRenderPass()
{
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = swapChainFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment = {};
    depthAttachment.format = findSupportedFormat(physicalDevice,
    { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
    VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef = {};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;


    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkAttachmentDescription attachments[] = {colorAttachment, depthAttachment};
    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = countof(attachments);
    renderPassInfo.pAttachments = attachments;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
        throw std::runtime_error("failed to create render pass!");
    }
}

void VulkanBaseApp::createDescriptorSetLayout()
{
    VkDescriptorSetLayoutBinding uboLayoutBinding = {};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.pImmutableSamplers = nullptr;
    //uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    //myline 
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

VkShaderModule createShaderModule(VkDevice device, const char *filename)
{
    std::vector<char> shaderContents;
    std::ifstream shaderFile(filename, std::ios_base::in | std::ios_base::binary);
    VkShaderModuleCreateInfo createInfo = {};
    VkShaderModule shaderModule;
    
    if (!shaderFile.good()) {
        throw std::runtime_error("Failed to load shader contents");
    }
    readFile(shaderFile, shaderContents);

    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = shaderContents.size();
 
    
    createInfo.pCode = reinterpret_cast<const uint32_t *>(shaderContents.data());

    if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }

    return shaderModule;
}

void VulkanBaseApp::getVertexDescriptions_1(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc)
{
}

void VulkanBaseApp::getVertexDescriptionsone(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc)
{
}



void VulkanBaseApp::getVertexDescriptions_2(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc)
{
}

void VulkanBaseApp::getVertexDescriptionstwo(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc)
{
}

void VulkanBaseApp::getVertexDescriptions_3(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc)
{
}



void VulkanBaseApp::getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info)
{

}

void VulkanBaseApp::createGraphicsPipeline()
{
    
    std::vector<VkPipelineShaderStageCreateInfo> shaderStageInfos(shaderFiles_1.size());
    for (size_t i = 0; i < shaderFiles_1.size(); i++) {
        shaderStageInfos[i] = {};
        shaderStageInfos[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageInfos[i].stage = shaderFiles_1[i].first;
        shaderStageInfos[i].module = createShaderModule(device, shaderFiles_1[i].second.c_str());
        shaderStageInfos[i].pName = "main";
        
    }

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};

    std::vector<VkVertexInputBindingDescription> vertexBindingDescriptions_1;
    std::vector<VkVertexInputAttributeDescription> vertexAttributeDescriptions_1;

    getVertexDescriptions_1(vertexBindingDescriptions_1, vertexAttributeDescriptions_1);

    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexBindingDescriptions_1.size());
    vertexInputInfo.pVertexBindingDescriptions = vertexBindingDescriptions_1.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptions_1.size());
    vertexInputInfo.pVertexAttributeDescriptions = vertexAttributeDescriptions_1.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    getAssemblyStateInfo(inputAssembly);


    ///////////////////////////////
    
   
    viewports.resize(vpcount);
    scissors.resize(vpcount);

    viewports[0] = { 200, 0, 500.0, 500.0, 0.0f, 1.0f };
    scissors[0] = {{ 200, 0 },{500,500}};
    viewports[1] = { 600, 500, 700, 700, 0.0f, 1.0f };
    scissors[1] = {{ 600, 500 },{700,700}};
    viewports[4] = { 1000, 1100, 900, 900, 0.0f, 1.0f };
    scissors[4] = {{ 1000, 1100 },{900,900}};
    viewports[3] = { 1200, 0, 500, 500, 0.0f, 1.0f };
    scissors[3] = {{ 1200, 0 },{500,500}};

    viewports[2] = { 0, 1100, 900, 900, 0.0f, 1.0f };
    scissors[2] = {{ 0, 1100 },{900,900}};


    VkPipelineViewportStateCreateInfo viewportState = {};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewports[0];
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissors[0];

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.lineWidth = 2.0f;
    rasterizer.polygonMode = VK_POLYGON_MODE_POINT;
    
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.minSampleShading = 1.0f; // Optional
    multisampling.pSampleMask = nullptr; // Optional
    multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
    multisampling.alphaToOneEnable = VK_FALSE; // Optional

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;


    ////////////////////////Push Constants//////////////////////////////////
    VkPushConstantRange push_constant;
    push_constant.offset = 0;
    push_constant.size = 16;
    push_constant.stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT;


    ////////////////////////////////////////////////////////////////////////


    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1; // Optional
    pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout; // Optional
    pipelineLayoutInfo.pushConstantRangeCount = 1; // Optional
    pipelineLayoutInfo.pPushConstantRanges = &push_constant; // Optional

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = static_cast<uint32_t>(shaderStageInfos.size());
    pipelineInfo.pStages = shaderStageInfos.data();

    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pDepthStencilState = &depthStencil; // Optional
    pipelineInfo.pColorBlendState = &colorBlending;
    // pipelineInfo.pDynamicState = &vpdsci ; // Optional

    pipelineInfo.layout = pipelineLayout;

    pipelineInfo.renderPass = renderPass;
    pipelineInfo.subpass = 0;

    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
    pipelineInfo.basePipelineIndex = -1; // Optional

    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipelines.graphicsPipeline_1) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    for (size_t i = 0; i < shaderStageInfos.size(); i++) {
        vkDestroyShaderModule(device, shaderStageInfos[i].module, nullptr);
    }

    //////////////////////////////////////pipelineone///////////////////////////////////////////////

    for (size_t i = 0; i < shaderFilesone.size(); i++) {

        shaderStageInfos[i].module = createShaderModule(device, shaderFilesone[i].second.c_str());

    }


    std::vector<VkVertexInputBindingDescription> vertexBindingDescriptionsone;
    std::vector<VkVertexInputAttributeDescription> vertexAttributeDescriptionsone;
    getVertexDescriptionsone(vertexBindingDescriptionsone, vertexAttributeDescriptionsone);

    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexBindingDescriptionsone.size());
    vertexInputInfo.pVertexBindingDescriptions = vertexBindingDescriptionsone.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptionsone.size());
    vertexInputInfo.pVertexAttributeDescriptions = vertexAttributeDescriptionsone.data();

    rasterizer.lineWidth = 1.0f;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    viewportState.pViewports = &viewports[3];
    viewportState.pScissors = &scissors[3];
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipelines.graphicsPipelineone) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline one!");
    }

    for (size_t i = 0; i < shaderStageInfos.size(); i++) {
        vkDestroyShaderModule(device, shaderStageInfos[i].module, nullptr);
    }
    
    // // ///////////////////////////pipeline_2/////////////////////////////////
    for (size_t i = 0; i < shaderFiles_2.size(); i++) {

        shaderStageInfos[i].module = createShaderModule(device, shaderFiles_2[i].second.c_str());
        // printf("%s \n",shaderFiles_2[i].second.c_str());

    }

    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewports[1];
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissors[1];    
    std::vector<VkVertexInputBindingDescription> vertexBindingDescriptions_2;
    std::vector<VkVertexInputAttributeDescription> vertexAttributeDescriptions_2;
    getVertexDescriptions_2(vertexBindingDescriptions_2, vertexAttributeDescriptions_2);
    
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexBindingDescriptions_2.size());
    vertexInputInfo.pVertexBindingDescriptions = vertexBindingDescriptions_2.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptions_2.size());
    vertexInputInfo.pVertexAttributeDescriptions = vertexAttributeDescriptions_2.data();

    rasterizer.lineWidth = 1.0f;
    rasterizer.polygonMode = VK_POLYGON_MODE_POINT;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipelines.graphicsPipeline_2) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline two!");
    }
  
    for (size_t i = 0; i < shaderStageInfos.size(); i++) {
        vkDestroyShaderModule(device, shaderStageInfos[i].module, nullptr);
    }

    // ////////////////////////////////////////////////////////////////////////////

 
    // ///////////////////////////////pipelinetwo/////////////////////////////////////
    for (size_t i = 0; i < shaderFilestwo.size(); i++) {

        shaderStageInfos[i].module = createShaderModule(device, shaderFilestwo[i].second.c_str());

    }

    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewports[4];
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissors[4];    
    
    std::vector<VkVertexInputBindingDescription> vertexBindingDescriptionstwo;
    std::vector<VkVertexInputAttributeDescription> vertexAttributeDescriptionstwo;
    getVertexDescriptionstwo(vertexBindingDescriptionstwo, vertexAttributeDescriptionstwo);
    
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexBindingDescriptionstwo.size());
    vertexInputInfo.pVertexBindingDescriptions = vertexBindingDescriptionstwo.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptionstwo.size());
    vertexInputInfo.pVertexAttributeDescriptions = vertexAttributeDescriptionstwo.data();

    rasterizer.lineWidth = 1.0f;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipelines.graphicsPipelinetwo) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline nine!");
    }
  
    for (size_t i = 0; i < shaderStageInfos.size(); i++) {
        vkDestroyShaderModule(device, shaderStageInfos[i].module, nullptr);
    }


    // //////////////////////////////////////////////////////////////////////////////

    //  // // ///////////////////////////pipeline_3/////////////////////////////////
    for (size_t i = 0; i < shaderFiles_3.size(); i++) {

        shaderStageInfos[i].module = createShaderModule(device, shaderFiles_3[i].second.c_str());

    }

    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewports[2];
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissors[2];    
    std::vector<VkVertexInputBindingDescription> vertexBindingDescriptions_3;
    std::vector<VkVertexInputAttributeDescription> vertexAttributeDescriptions_3;
    getVertexDescriptions_3(vertexBindingDescriptions_3, vertexAttributeDescriptions_3);
    
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexBindingDescriptions_3.size());
    vertexInputInfo.pVertexBindingDescriptions = vertexBindingDescriptions_3.data();
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptions_3.size());
    vertexInputInfo.pVertexAttributeDescriptions = vertexAttributeDescriptions_3.data();

    rasterizer.lineWidth = 1.0f;
    rasterizer.polygonMode = VK_POLYGON_MODE_POINT;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipelines.graphicsPipeline_3) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline four!");
    }
  
    for (size_t i = 0; i < shaderStageInfos.size(); i++) {
        vkDestroyShaderModule(device, shaderStageInfos[i].module, nullptr);
    }

    // ////////////////////////////////////////////////////////////////////////////

   

   
  


}

void VulkanBaseApp::createFramebuffers()
{
    swapChainFramebuffers.resize(swapChainImageViews.size());
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        VkImageView attachments[] = {
            swapChainImageViews[i],
            depthImageView
        };

        VkFramebufferCreateInfo framebufferInfo = {};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = renderPass;
        framebufferInfo.attachmentCount = countof(attachments);
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = swapChainExtent.width;
        framebufferInfo.height = swapChainExtent.height;
        framebufferInfo.layers = 1;
        

        if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void VulkanBaseApp::createCommandPool()
{
    VkCommandPoolCreateInfo poolInfo = {};
    uint32_t graphicsIndex, presentIndex;

    findGraphicsQueueIndicies(physicalDevice, surface, graphicsIndex, presentIndex);

    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = graphicsIndex;
    poolInfo.flags = 0; // Optional

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create command pool!");
    }
}

static void transitionImageLayout(VulkanBaseApp *app, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkCommandBuffer commandBuffer = app->beginSingleTimeCommands();

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;

    if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

        if (format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT) {
            barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
        }
    }
    else {
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags sourceStage;
    VkPipelineStageFlags destinationStage;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }
    else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    }
    else {
        throw std::invalid_argument("unsupported layout transition!");
    }

    vkCmdPipelineBarrier(
        commandBuffer,
        sourceStage, destinationStage,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );

    app->endSingleTimeCommands(commandBuffer);
}

void VulkanBaseApp::createDepthResources()
{
    VkFormat depthFormat = findSupportedFormat(physicalDevice,
    { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
    VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    createImage(physicalDevice, device, swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
    depthImageView = createImageView(device, depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);
    transitionImageLayout(this, depthImage, depthFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
}

void VulkanBaseApp::createUniformBuffers()
{
    VkDeviceSize size = getUniformSize();
    if (size > 0) {
        uniformBuffers.resize(swapChainImages.size());
        uniformMemory.resize(swapChainImages.size());
        for (size_t i = 0; i < uniformBuffers.size(); i++) {
            createBuffer(getUniformSize(),
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         uniformBuffers[i], uniformMemory[i]);
        }
    }
}

void VulkanBaseApp::createDescriptorPool()
{
    VkDescriptorPoolSize poolSize = {};
    poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSize.descriptorCount = static_cast<uint32_t>(swapChainImages.size());
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = static_cast<uint32_t>(swapChainImages.size());;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void VulkanBaseApp::createDescriptorSets()
{
    std::vector<VkDescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainImages.size());
    allocInfo.pSetLayouts = layouts.data();
    descriptorSets.resize(swapChainImages.size());

    if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    VkDescriptorBufferInfo bufferInfo = {};
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet descriptorWrite = {};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.dstBinding = 0;
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &bufferInfo;
    descriptorWrite.pImageInfo = nullptr; // Optional
    descriptorWrite.pTexelBufferView = nullptr; // Optional

    for (size_t i = 0; i < swapChainImages.size(); i++) {
        bufferInfo.buffer = uniformBuffers[i];
        descriptorWrite.dstSet = descriptorSets[i];
        vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
    }
}


void VulkanBaseApp::createCommandBuffers()
{
    
    commandBuffers.resize(swapChainFramebuffers.size());
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

    VkResult res = vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data());

    if (res != VK_SUCCESS) {
        
        throw std::runtime_error("failed to allocate command buffers!");
    }

    for (size_t i = 0; i < commandBuffers.size(); i++) {
        VkCommandBufferBeginInfo beginInfo = {};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
        beginInfo.pInheritanceInfo = nullptr; // Optional

        if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[i];

        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        VkClearValue clearColors[2];
        clearColors[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
        clearColors[1].depthStencil = { 1.0f, 0 };
        renderPassInfo.clearValueCount = countof(clearColors);
        renderPassInfo.pClearValues = clearColors;

        vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
       
		vkCmdSetViewport(commandBuffers[i], 0,1,&viewports[0]);
      
        vkCmdSetScissor(commandBuffers[i],0,1,&scissors[0]);
       
        vkCmdSetLineWidth(commandBuffers[i],2.0);
        
        vkCmdBindDescriptorSets(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);
        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.graphicsPipeline_1);
        fillRenderingCommandBuffer_1(commandBuffers[i]);
        
        vkCmdSetViewport(commandBuffers[i], 0,1,&viewports[3]);            
        vkCmdSetScissor(commandBuffers[i],0,1,&scissors[3]);
        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.graphicsPipelineone);
        fillRenderingCommandBufferone(commandBuffers[i]);
 

        vkCmdSetViewport(commandBuffers[i], 0,1,&viewports[1]);            
        vkCmdSetScissor(commandBuffers[i],0,1,&scissors[1]);
        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.graphicsPipeline_2);
        fillRenderingCommandBuffer_2(commandBuffers[i]);

        vkCmdSetViewport(commandBuffers[i], 0,1,&viewports[4]);            
        vkCmdSetScissor(commandBuffers[i],0,1,&scissors[4]);
        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.graphicsPipelinetwo);
        fillRenderingCommandBuffertwo(commandBuffers[i]);


        vkCmdSetViewport(commandBuffers[i], 0,1,&viewports[2]);            
        vkCmdSetScissor(commandBuffers[i],0,1,&scissors[2]);
        vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.graphicsPipeline_3);
        fillRenderingCommandBuffer_3(commandBuffers[i]);



        vkCmdEndRenderPass(commandBuffers[i]);

        if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }
    }
}



void VulkanBaseApp::createSyncObjects()
{
    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image available semaphore!");
        }
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image available semaphore!");
        }
        if (vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create image available semaphore!");
        }
        
    }
}

void VulkanBaseApp::getWaitFrameSemaphores(std::vector<VkSemaphore>& wait, std::vector<VkPipelineStageFlags>& waitStages) const
{
}

void VulkanBaseApp::getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const
{
}

VkDeviceSize VulkanBaseApp::getUniformSize() const
{
    return VkDeviceSize(0);
}

void VulkanBaseApp::updateUniformBuffer(uint32_t imageIndex, bool shift)
{
}

void VulkanBaseApp::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void VulkanBaseApp::createExternalBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkExternalMemoryHandleTypeFlagsKHR extMemHandleType, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VkExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR = {};
    vulkanExportMemoryAllocateInfoKHR.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
 
    VkExternalMemoryBufferCreateInfo vulkanExternalMemoryBufferCreateInfo ={};
    vulkanExternalMemoryBufferCreateInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
   
    vulkanExportMemoryAllocateInfoKHR.pNext = NULL;
    vulkanExportMemoryAllocateInfoKHR.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    vulkanExternalMemoryBufferCreateInfo.pNext = NULL;
    vulkanExternalMemoryBufferCreateInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    bufferInfo.pNext = &vulkanExternalMemoryBufferCreateInfo;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo = {};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &vulkanExportMemoryAllocateInfoKHR;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate external buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void *VulkanBaseApp::getMemHandle(VkDeviceMemory memory, VkExternalMemoryHandleTypeFlagBits handleType)
{

    int fd = -1;

    VkMemoryGetFdInfoKHR vkMemoryGetFdInfoKHR = {};
    vkMemoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
    vkMemoryGetFdInfoKHR.pNext = NULL;
    vkMemoryGetFdInfoKHR.memory = memory;
    vkMemoryGetFdInfoKHR.handleType = handleType;

    PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR;
    fpGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
    if (!fpGetMemoryFdKHR) {
        throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
    }
    if (fpGetMemoryFdKHR(device, &vkMemoryGetFdInfoKHR, &fd) != VK_SUCCESS) {
        throw std::runtime_error("Failed to retrieve handle for buffer!");
    }
    return (void *)(uintptr_t)fd;

}

void *VulkanBaseApp::getSemaphoreHandle(VkSemaphore semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType)
{

    int fd;

    VkSemaphoreGetFdInfoKHR semaphoreGetFdInfoKHR = {};
    semaphoreGetFdInfoKHR.sType =VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    semaphoreGetFdInfoKHR.pNext = NULL;
    semaphoreGetFdInfoKHR.semaphore = semaphore;
    semaphoreGetFdInfoKHR.handleType = handleType;

    PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR;
    fpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
    if (!fpGetSemaphoreFdKHR) {
        throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
    }
    if (fpGetSemaphoreFdKHR(device, &semaphoreGetFdInfoKHR, &fd) != VK_SUCCESS) {
        throw std::runtime_error("Failed to retrieve handle for buffer!");
    }

    return (void *)(uintptr_t)fd;
}

void VulkanBaseApp::createExternalSemaphore(VkSemaphore& semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType)
{
    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkExportSemaphoreCreateInfoKHR exportSemaphoreCreateInfo = {};
    exportSemaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;

    exportSemaphoreCreateInfo.pNext = NULL;

    exportSemaphoreCreateInfo.handleTypes = handleType;
    semaphoreInfo.pNext = &exportSemaphoreCreateInfo;

    if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &semaphore) != VK_SUCCESS) {
        throw std::runtime_error("failed to create synchronization objects for a CUDA-Vulkan!");
    }
}

void VulkanBaseApp::importExternalBuffer(void *handle, VkExternalMemoryHandleTypeFlagBits handleType, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory)
{
    VkBufferCreateInfo bufferInfo = {};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);


    VkImportMemoryFdInfoKHR handleInfo = {};
    handleInfo.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR;
    handleInfo.pNext = NULL;
    handleInfo.fd = (int)(uintptr_t)handle;
    handleInfo.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkMemoryAllocateInfo memAllocation = {};
    memAllocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAllocation.pNext = (void *)&handleInfo;
    memAllocation.allocationSize = size;
    memAllocation.memoryTypeIndex = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &memAllocation, nullptr, &memory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to import allocation!");
    }

    vkBindBufferMemory(device, buffer, memory, 0);
}

void VulkanBaseApp::copyBuffer(VkBuffer dst, VkBuffer src, VkDeviceSize srcOffset,VkDeviceSize size)
{

    VkCommandBuffer commandBuffer = beginSingleTimeCommands();

    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = srcOffset;
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, src, dst, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
}

void VulkanBaseApp::updatecommandBuffers()
{
    createCommandBuffers();
}

void VulkanBaseApp::drawFrame(bool shift)
{
    
    size_t currentFrameIdx = currentFrame % MAX_FRAMES_IN_FLIGHT;
   
    vkWaitForFences(device, 1, &inFlightFences[currentFrameIdx], VK_TRUE, std::numeric_limits<uint64_t>::max());
    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(device, swapChain, std::numeric_limits<uint64_t>::max(), imageAvailableSemaphores[currentFrameIdx], VK_NULL_HANDLE, &imageIndex);
    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        
        recreateSwapChain();
    }
    else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("Failed to acquire swap chain image!");
    }
   
    updateUniformBuffer(imageIndex,shift);

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    std::vector<VkSemaphore> waitSemaphores;
    std::vector<VkPipelineStageFlags> waitStages;
    
    waitSemaphores.push_back(imageAvailableSemaphores[currentFrameIdx]);
    waitStages.push_back(VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);
    getWaitFrameSemaphores(waitSemaphores, waitStages);

    submitInfo.waitSemaphoreCount = (uint32_t)waitSemaphores.size();
    submitInfo.pWaitSemaphores = waitSemaphores.data();
    submitInfo.pWaitDstStageMask = waitStages.data();


    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffers[imageIndex];

    std::vector<VkSemaphore> signalSemaphores;
    getSignalFrameSemaphores(signalSemaphores);
    signalSemaphores.push_back(renderFinishedSemaphores[currentFrameIdx]);
    submitInfo.signalSemaphoreCount = (uint32_t)signalSemaphores.size();
    submitInfo.pSignalSemaphores = signalSemaphores.data();
    

    vkResetFences(device, 1, &inFlightFences[currentFrameIdx]);

    VkResult res = vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrameIdx]);
    if  (res != VK_SUCCESS) {
        
        throw std::runtime_error("failed to submit draw command buffer!");
    }
    

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &renderFinishedSemaphores[currentFrameIdx];

    VkSwapchainKHR swapChains[] = { swapChain };
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;
    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(presentQueue, &presentInfo);
   
    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
        recreateSwapChain();
        framebufferResized = false;
    }
    else if (result != VK_SUCCESS) {
        throw std::runtime_error("Failed to acquire swap chain image!");
    }
      
    currentFrame++;
  
   
}




void VulkanBaseApp::cleanupSwapChain()
{

    if (depthImageView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, depthImageView, nullptr);
    }
    if (depthImage != VK_NULL_HANDLE) {
        vkDestroyImage(device, depthImage, nullptr);
    }
    if (depthImageMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, depthImageMemory, nullptr);
    }

    for (size_t i = 0; i < uniformBuffers.size(); i++) {
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformMemory[i], nullptr);
    }

    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    }

    for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
        vkDestroyFramebuffer(device, swapChainFramebuffers[i], nullptr);
    }

    if (pipelines.graphicsPipeline_1 != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipelines.graphicsPipeline_1, nullptr);
    }

    if (pipelines.graphicsPipelineone != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipelines.graphicsPipelineone, nullptr);
    }

    if (pipelines.graphicsPipeline_2 != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipelines.graphicsPipeline_2, nullptr);
    }

    if (pipelines.graphicsPipelinetwo != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipelines.graphicsPipelinetwo, nullptr);
    }

    if (pipelines.graphicsPipeline_3 != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, pipelines.graphicsPipeline_3, nullptr);
    }


    if (pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    }

    if (renderPass != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device, renderPass, nullptr);
    }

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        vkDestroyImageView(device, swapChainImageViews[i], nullptr);
    }

    if (swapChain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }
}

void VulkanBaseApp::recreateSwapChain()
{
    int width, height;

    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwWaitEvents();
        glfwGetFramebufferSize(window, &width, &height);
    }

    vkDeviceWaitIdle(device);
    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createGraphicsPipeline();
    createDepthResources();
    createFramebuffers();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
}

void VulkanBaseApp::mainLoop(bool shift)

{

    while (!glfwWindowShouldClose(window) ) {
        glfwPollEvents();
        drawFrame(shift);
        
    }
    std::cout<<"Mainloop Terminated \n"<<std::endl;
    vkDeviceWaitIdle(device);
    std::cout<<"Exiting Appplication\n"<<std::endl;
}

void readFile(std::istream& s, std::vector<char>& data)
{
    s.seekg(0, std::ios_base::end);
    data.resize(s.tellg());
    s.clear();
    s.seekg(0, std::ios_base::beg);
    s.read(data.data(), data.size());
}
