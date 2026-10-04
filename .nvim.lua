vim.filetype.add({
  extension = {
    hlsl = 'hlsl',
    comp = 'hlsl',
    vert = 'hlsl',
    frag = 'hlsl',
  },
})

vim.lsp.config('slangd', {
  settings = {
    slang = {
      predefinedMacros = { 'MINIDXNN_NO_INCLUDE_DX_LINALG' },
      additionalSearchPaths = { vim.fs.joinpath(vim.fn.getcwd(), 'include/minidxnn/hlsl') },
    },
  },
})

vim.lsp.enable('clangd')
vim.lsp.enable('cmake')
vim.lsp.enable('slangd')
