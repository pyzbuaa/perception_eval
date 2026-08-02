import type { CategoryDefinition } from './types'

function categoryFromValue(value: unknown): CategoryDefinition {
  if (!value || typeof value !== 'object') throw new Error('类别条目必须是对象')
  const item = value as Record<string, unknown>
  const id = Number(item.id ?? item.category_id)
  const name = String(item.name ?? item.category_name ?? '').trim()
  if (!Number.isInteger(id) || id < 0 || !name) throw new Error('类别 ID 必须是非负整数，名称不能为空')
  return { id, name }
}

function trimCsvValue(value: string) {
  const trimmed = value.trim()
  return trimmed.startsWith('"') && trimmed.endsWith('"')
    ? trimmed.slice(1, -1).replace(/""/g, '"').trim()
    : trimmed
}

function parseDelimitedCategories(text: string): CategoryDefinition[] {
  return text.split(/\r?\n/).flatMap((rawLine) => {
    const line = rawLine.trim()
    if (!line || line.startsWith('#')) return []
    const separator = line.includes(',') ? ',' : line.includes('\t') ? '\t' : null
    const parts = separator ? line.split(separator) : line.match(/^(\S+)\s+(.+)$/)?.slice(1) || []
    const first = trimCsvValue(parts.shift() || '')
    const second = trimCsvValue(parts.join(separator || ' '))
    if (['id', 'category_id'].includes(first.toLocaleLowerCase())) return []
    return [categoryFromValue({ id: first, name: second })]
  })
}

export function parseCategoryFile(text: string, fileName: string): CategoryDefinition[] {
  const content = text.replace(/^\uFEFF/, '').trim()
  if (!content) throw new Error('类别文件为空')
  let categories: CategoryDefinition[]
  if (fileName.toLocaleLowerCase().endsWith('.json') || content.startsWith('[') || content.startsWith('{')) {
    const parsed = JSON.parse(content) as unknown
    const values = Array.isArray(parsed)
      ? parsed
      : parsed && typeof parsed === 'object' && Array.isArray((parsed as Record<string, unknown>).categories)
        ? (parsed as { categories: unknown[] }).categories
        : null
    if (!values) throw new Error('JSON 应为类别数组，或包含 categories 数组')
    categories = values.map(categoryFromValue)
  } else {
    categories = parseDelimitedCategories(content)
  }
  const ids = categories.map((item) => item.id)
  const names = categories.map((item) => item.name.toLocaleLowerCase())
  if (!categories.length) throw new Error('类别文件中没有有效类别')
  if (new Set(ids).size !== ids.length) throw new Error('类别 ID 不能重复')
  if (new Set(names).size !== names.length) throw new Error('类别名称不能重复')
  return categories
}
