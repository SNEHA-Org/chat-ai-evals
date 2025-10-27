import React, { useMemo } from 'react';
import {
  useReactTable,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  flexRender,
  createColumnHelper,
  type ColumnDef,
} from '@tanstack/react-table';
import { ChevronLeft, ChevronRight, Search, FileText } from 'lucide-react';
import type { GoldenDataset } from '../hooks/useSocket';

interface GoldenDatasetViewerProps {
  dataset: GoldenDataset | null;
}

export const GoldenDatasetViewer: React.FC<GoldenDatasetViewerProps> = ({ dataset }) => {
  const [globalFilter, setGlobalFilter] = React.useState('');

  const columns = useMemo<ColumnDef<Record<string, any>>[]>(() => {
    if (!dataset) return [];

    const columnHelper = createColumnHelper<Record<string, any>>();

    return dataset.columns.map((column) =>
      columnHelper.accessor(column, {
        header: column,
        cell: (info) => {
          const value = info.getValue() as string;
          if (!value) return <span className="text-gray-400">—</span>;
          
          // Truncate long text for better display
          const truncated = value.length > 100 ? value.substring(0, 100) + '...' : value;
          return (
            <div 
              title={value} 
              className="max-w-xs overflow-hidden text-ellipsis whitespace-nowrap"
            >
              {truncated}
            </div>
          );
        },
      })
    );
  }, [dataset]);

  const table = useReactTable({
    data: dataset?.rows || [],
    columns,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
    state: {
      globalFilter,
    },
    onGlobalFilterChange: setGlobalFilter,
    initialState: {
      pagination: {
        pageSize: 5,
      },
    },
  });

  if (!dataset) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-gray-500">
        <FileText className="h-12 w-12 mb-4" />
        <p className="text-lg font-medium">No dataset loaded</p>
        <p className="text-sm">Click "Refresh" to load the golden dataset</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Dataset Info */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-gray-600">
          <span className="font-medium">{dataset.total_count}</span> examples in{' '}
          <span className="font-medium">{dataset.filename}</span>
        </div>
        
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <input
            value={globalFilter ?? ''}
            onChange={(e) => setGlobalFilter(e.target.value)}
            placeholder="Search examples..."
            className="pl-9 pr-4 py-2 border border-gray-300 rounded-md text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto border border-gray-200 rounded-lg">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            {table.getHeaderGroups().map((headerGroup) => (
              <tr key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <th
                    key={header.id}
                    className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer hover:bg-gray-100"
                    onClick={header.column.getToggleSortingHandler()}
                  >
                    <div className="flex items-center space-x-1">
                      <span>
                        {header.isPlaceholder
                          ? null
                          : flexRender(header.column.columnDef.header, header.getContext())}
                      </span>
                      <span className="text-gray-400">
                        {{
                          asc: '↑',
                          desc: '↓',
                        }[header.column.getIsSorted() as string] ?? null}
                      </span>
                    </div>
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {table.getRowModel().rows.map((row) => (
              <tr key={row.id} className="hover:bg-gray-50">
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id} className="px-4 py-3 text-sm text-gray-900">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between">
        <div className="text-sm text-gray-700">
          Showing {table.getState().pagination.pageIndex * table.getState().pagination.pageSize + 1} to{' '}
          {Math.min(
            (table.getState().pagination.pageIndex + 1) * table.getState().pagination.pageSize,
            table.getFilteredRowModel().rows.length
          )}{' '}
          of {table.getFilteredRowModel().rows.length} entries
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
            className="p-2 border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          
          <span className="text-sm text-gray-700">
            Page {table.getState().pagination.pageIndex + 1} of {table.getPageCount()}
          </span>
          
          <button
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
            className="p-2 border border-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
};