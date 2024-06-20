
#pragma once
#ifndef __VULKANBASEAPP_H__
#define __VULKANBASEAPP_H__

#include <string>
#include <vector>
#include <vulkan/vulkan.h>

struct GLFWwindow;
typedef float fvec4[4];

class VulkanBaseApp
{
public:

    VulkanBaseApp(const std::string& appName, bool enableValidation = false);
    static VkExternalSemaphoreHandleTypeFlagBits getDefaultSemaphoreHandleType();
    static VkExternalMemoryHandleTypeFlagBits getDefaultMemHandleType();
    virtual ~VulkanBaseApp();
    void init();
    void *getMemHandle(VkDeviceMemory memory, VkExternalMemoryHandleTypeFlagBits handleType);
    void *getSemaphoreHandle(VkSemaphore semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType);
    void createExternalSemaphore(VkSemaphore& semaphore, VkExternalSemaphoreHandleTypeFlagBits handleType);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void createExternalBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkExternalMemoryHandleTypeFlagsKHR extMemHandleType, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void importExternalBuffer(void *handle, VkExternalMemoryHandleTypeFlagBits handleType, size_t size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& memory);
    void copyBuffer(VkBuffer dst, VkBuffer src, VkDeviceSize srcOffset,VkDeviceSize size);
    VkCommandBuffer beginSingleTimeCommands();
    void endSingleTimeCommands(VkCommandBuffer commandBuffer);
    void mainLoop(bool shift);

protected:

    const std::string appName;
    const bool enableValidation;
    VkInstance instance;
    VkDebugUtilsMessengerEXT debugMessenger;
    VkSurfaceKHR surface;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<std::pair<VkShaderStageFlagBits, std::string> > shaderFiles_1;
    std::vector<std::pair<VkShaderStageFlagBits, std::string> > shaderFilesone;
    std::vector<std::pair<VkShaderStageFlagBits, std::string> > shaderFiles_2;
    std::vector<std::pair<VkShaderStageFlagBits, std::string> > shaderFilestwo;
    std::vector<std::pair<VkShaderStageFlagBits, std::string> > shaderFiles_3;
   
  
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    std::vector<VkFramebuffer> swapChainFramebuffers;
    int vpcount;
    std::vector< VkViewport> viewports ;
    std::vector<VkRect2D> scissors;
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformMemory;
    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;
    size_t currentFrame;
    bool framebufferResized;
    uint8_t  vkDeviceUUID[VK_UUID_SIZE];
    

    virtual void initVulkanApp() {}
    virtual void fillRenderingCommandBuffer_1(VkCommandBuffer& buffer) {};
    virtual void fillRenderingCommandBuffer_2(VkCommandBuffer& buffer) {};
    virtual void fillRenderingCommandBuffer_3(VkCommandBuffer& buffer) {};
    virtual void fillRenderingCommandBufferone(VkCommandBuffer& buffer) {};
    virtual void fillRenderingCommandBuffertwo(VkCommandBuffer& buffer) {};

    virtual std::vector<const char *> getRequiredExtensions() const;
    virtual std::vector<const char *> getRequiredDeviceExtensions() const;
    virtual void getVertexDescriptions_1(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc);
    virtual void getVertexDescriptionsone(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc);
    virtual void getVertexDescriptions_2(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc);
    virtual void getVertexDescriptionstwo(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc);
    virtual void getVertexDescriptions_3(std::vector<VkVertexInputBindingDescription>& bindingDesc, std::vector<VkVertexInputAttributeDescription>& attribDesc);
    
    virtual void getAssemblyStateInfo(VkPipelineInputAssemblyStateCreateInfo& info);
    virtual void getWaitFrameSemaphores(std::vector<VkSemaphore>& wait, std::vector< VkPipelineStageFlags>& waitStages) const;
    virtual void getSignalFrameSemaphores(std::vector<VkSemaphore>& signal) const;
    virtual VkDeviceSize getUniformSize() const;
    virtual void updateUniformBuffer(uint32_t imageIndex, bool shift);
    virtual void updatecommandBuffers();
    virtual void drawFrame(bool shift);
    GLFWwindow *window;

private:
    
    void initWindow();
    void initVulkan();
    void createInstance();
    void createSurface();
    void createDevice();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createDepthResources();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createCommandBuffers();
    void createSyncObjects();

    void cleanupSwapChain();
    void recreateSwapChain();

    bool isSuitableDevice(VkPhysicalDevice dev) const;
    static void resizeCallback(GLFWwindow *window, int width, int height);
};

void readFile(std::istream& s, std::vector<char>& data);

#endif /* __VULKANBASEAPP_H__ */
