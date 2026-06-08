/**
 * 聊天文件上传 composable
 * 从 ChatView.vue 中提取的文件上传/粘贴/附件管理逻辑
 */
import { ref, reactive } from 'vue'
import { API_BASE } from './useApi.js'

export function useChatUpload() {
  const chatFileInput = ref(null)
  const chatAttachments = reactive([])

  const onChatFileSelect = async (e) => {
    const files = Array.from(e.target.files || [])
    if (e.target) e.target.value = ''
    await uploadChatFiles(files)
  }

  const onPaste = async (e) => {
    const items = Array.from(e.clipboardData?.items || [])
    const imageFiles = []
    for (const item of items) {
      if (item.type.startsWith('image/')) {
        const blob = item.getAsFile()
        if (blob) {
          const ext = item.type.split('/')[1] || 'png'
          const file = new File([blob], `paste_${Date.now()}.${ext}`, { type: item.type })
          imageFiles.push(file)
        }
      }
    }
    if (imageFiles.length) {
      e.preventDefault()
      await uploadChatFiles(imageFiles)
    }
  }

  const uploadChatFiles = async (files) => {
    for (const file of files) {
      const ext = (file.name || '').split('.').pop()?.toLowerCase() || ''
      const isImage = ['png','jpg','jpeg','bmp','gif','webp','tiff','tif'].includes(ext)
      const att = reactive({
        name: file.name,
        isImage,
        parsedText: '',
        uploading: true,
      })
      chatAttachments.push(att)
      try {
        const formData = new FormData()
        formData.append('file', file)
        const res = await fetch(`${API_BASE}/api/chat/upload`, { method: 'POST', body: formData })
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || '上传失败')
        const data = await res.json()
        att.parsedText = data.parsed_text || ''
        att.uploading = false
      } catch (err) {
        att.parsedText = `[上传失败: ${err.message}]`
        att.uploading = false
      }
    }
  }

  const removeChatAttachment = (idx) => {
    chatAttachments.splice(idx, 1)
  }

  const clearAttachments = () => {
    chatAttachments.splice(0, chatAttachments.length)
  }

  return {
    chatFileInput,
    chatAttachments,
    onChatFileSelect,
    onPaste,
    uploadChatFiles,
    removeChatAttachment,
    clearAttachments,
  }
}
