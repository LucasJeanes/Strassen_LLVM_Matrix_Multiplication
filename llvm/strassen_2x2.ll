; Strassen Matrix Multiplication LLVM IR Implementation
; Optimized for NxN matrices (N must be power of 2)

; LLVM Target and Data Layout (important for optimized IR based on host architecture)
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; External memory allocation functions
declare noalias ptr @malloc(i64) #0
declare void @free(ptr) #1
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, 
                                    ptr noalias nocapture readonly, i64, i1 immarg) #2

; noalias: Won't create  aliasing issues - optimization
; nounwind: Function won't throw exceptions
; nofree: Doesn't store pointers - memory safe optimization
; immarg: Argument is immediate (constant) - optimization
; Matrix structure: stored as flat array in row-major order
; Matrix[i][j] = data[i * size + j]

; Main Strassen multiplication function
define void @strassen_multiply(ptr %A, ptr %B, ptr %C, i32 %n) #3 {
entry:
  %n_i64 = zext i32 %n to i64
  %threshold = icmp ule i32 %n, 64
  br i1 %threshold, label %base_case, label %recursive_case

base_case:
  ; For small matrices, use standard multiplication
  call void @naive_multiply(ptr %A, ptr %B, ptr %C, i32 %n)
  ret void

recursive_case:
  ; Divide matrix into quadrants and apply Strassen's algorithm
  %half_n = lshr i32 %n, 1
  %half_n_i64 = zext i32 %half_n to i64
  %quarter_size = mul i64 %half_n_i64, %half_n_i64
  %quarter_bytes = mul i64 %quarter_size, 8
  
  ; Allocate memory for submatrices and intermediate results
  ; A11, A12, A21, A22, B11, B12, B21, B22
  %A11 = call ptr @malloc(i64 %quarter_bytes)
  %A12 = call ptr @malloc(i64 %quarter_bytes)
  %A21 = call ptr @malloc(i64 %quarter_bytes)
  %A22 = call ptr @malloc(i64 %quarter_bytes)
  %B11 = call ptr @malloc(i64 %quarter_bytes)
  %B12 = call ptr @malloc(i64 %quarter_bytes)
  %B21 = call ptr @malloc(i64 %quarter_bytes)
  %B22 = call ptr @malloc(i64 %quarter_bytes)
  
  ; Strassen's 7 products: P1 through P7
  %P1 = call ptr @malloc(i64 %quarter_bytes)
  %P2 = call ptr @malloc(i64 %quarter_bytes)
  %P3 = call ptr @malloc(i64 %quarter_bytes)
  %P4 = call ptr @malloc(i64 %quarter_bytes)
  %P5 = call ptr @malloc(i64 %quarter_bytes)
  %P6 = call ptr @malloc(i64 %quarter_bytes)
  %P7 = call ptr @malloc(i64 %quarter_bytes)
  
  ; Temporary matrices for additions/subtractions
  %temp1 = call ptr @malloc(i64 %quarter_bytes)
  %temp2 = call ptr @malloc(i64 %quarter_bytes)
  
  ; Extract quadrants from A and B
  call void @extract_quadrant(ptr %A, ptr %A11, i32 %n, i32 %half_n, i32 0, i32 0)
  call void @extract_quadrant(ptr %A, ptr %A12, i32 %n, i32 %half_n, i32 0, i32 %half_n)
  call void @extract_quadrant(ptr %A, ptr %A21, i32 %n, i32 %half_n, i32 %half_n, i32 0)
  call void @extract_quadrant(ptr %A, ptr %A22, i32 %n, i32 %half_n, i32 %half_n, i32 %half_n)
  
  call void @extract_quadrant(ptr %B, ptr %B11, i32 %n, i32 %half_n, i32 0, i32 0)
  call void @extract_quadrant(ptr %B, ptr %B12, i32 %n, i32 %half_n, i32 0, i32 %half_n)
  call void @extract_quadrant(ptr %B, ptr %B21, i32 %n, i32 %half_n, i32 %half_n, i32 0)
  call void @extract_quadrant(ptr %B, ptr %B22, i32 %n, i32 %half_n, i32 %half_n, i32 %half_n)
  
  ; Compute P1 = A11 * (B12 - B22)
  call void @matrix_subtract(ptr %B12, ptr %B22, ptr %temp1, i32 %half_n)
  call void @strassen_multiply(ptr %A11, ptr %temp1, ptr %P1, i32 %half_n)
  
  ; Compute P2 = (A11 + A12) * B22
  call void @matrix_add(ptr %A11, ptr %A12, ptr %temp1, i32 %half_n)
  call void @strassen_multiply(ptr %temp1, ptr %B22, ptr %P2, i32 %half_n)
  
  ; Compute P3 = (A21 + A22) * B11
  call void @matrix_add(ptr %A21, ptr %A22, ptr %temp1, i32 %half_n)
  call void @strassen_multiply(ptr %temp1, ptr %B11, ptr %P3, i32 %half_n)
  
  ; Compute P4 = A22 * (B21 - B11)
  call void @matrix_subtract(ptr %B21, ptr %B11, ptr %temp1, i32 %half_n)
  call void @strassen_multiply(ptr %A22, ptr %temp1, ptr %P4, i32 %half_n)
  
  ; Compute P5 = (A11 + A22) * (B11 + B22)
  call void @matrix_add(ptr %A11, ptr %A22, ptr %temp1, i32 %half_n)
  call void @matrix_add(ptr %B11, ptr %B22, ptr %temp2, i32 %half_n)
  call void @strassen_multiply(ptr %temp1, ptr %temp2, ptr %P5, i32 %half_n)
  
  ; Compute P6 = (A12 - A22) * (B21 + B22)
  call void @matrix_subtract(ptr %A12, ptr %A22, ptr %temp1, i32 %half_n)
  call void @matrix_add(ptr %B21, ptr %B22, ptr %temp2, i32 %half_n)
  call void @strassen_multiply(ptr %temp1, ptr %temp2, ptr %P6, i32 %half_n)
  
  ; Compute P7 = (A11 - A21) * (B11 + B12)
  call void @matrix_subtract(ptr %A11, ptr %A21, ptr %temp1, i32 %half_n)
  call void @matrix_add(ptr %B11, ptr %B12, ptr %temp2, i32 %half_n)
  call void @strassen_multiply(ptr %temp1, ptr %temp2, ptr %P7, i32 %half_n)
  
  ; Compute result quadrants
  ; C11 = P5 + P4 - P2 + P6
  call void @matrix_add(ptr %P5, ptr %P4, ptr %temp1, i32 %half_n)
  call void @matrix_subtract(ptr %temp1, ptr %P2, ptr %temp2, i32 %half_n)
  call void @matrix_add(ptr %temp2, ptr %P6, ptr %temp1, i32 %half_n)
  call void @insert_quadrant(ptr %temp1, ptr %C, i32 %n, i32 %half_n, i32 0, i32 0)
  
  ; C12 = P1 + P2
  call void @matrix_add(ptr %P1, ptr %P2, ptr %temp1, i32 %half_n)
  call void @insert_quadrant(ptr %temp1, ptr %C, i32 %n, i32 %half_n, i32 0, i32 %half_n)
  
  ; C21 = P3 + P4
  call void @matrix_add(ptr %P3, ptr %P4, ptr %temp1, i32 %half_n)
  call void @insert_quadrant(ptr %temp1, ptr %C, i32 %n, i32 %half_n, i32 %half_n, i32 0)
  
  ; C22 = P5 + P1 - P3 - P7
  call void @matrix_add(ptr %P5, ptr %P1, ptr %temp1, i32 %half_n)
  call void @matrix_subtract(ptr %temp1, ptr %P3, ptr %temp2, i32 %half_n)
  call void @matrix_subtract(ptr %temp2, ptr %P7, ptr %temp1, i32 %half_n)
  call void @insert_quadrant(ptr %temp1, ptr %C, i32 %n, i32 %half_n, i32 %half_n, i32 %half_n)
  
  ; Clean up memory
  call void @free(ptr %A11)
  call void @free(ptr %A12)
  call void @free(ptr %A21)
  call void @free(ptr %A22)
  call void @free(ptr %B11)
  call void @free(ptr %B12)
  call void @free(ptr %B21)
  call void @free(ptr %B22)
  call void @free(ptr %P1)
  call void @free(ptr %P2)
  call void @free(ptr %P3)
  call void @free(ptr %P4)
  call void @free(ptr %P5)
  call void @free(ptr %P6)
  call void @free(ptr %P7)
  call void @free(ptr %temp1)
  call void @free(ptr %temp2)
  
  ret void
}

; Naive matrix multiplication for base case
define void @naive_multiply(ptr %A, ptr %B, ptr %C, i32 %n) #3 {
entry:
  br label %i_loop

i_loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %i_next_block ]
  %i_cmp = icmp ult i32 %i, %n
  br i1 %i_cmp, label %j_loop_init, label %exit

j_loop_init:
  br label %j_loop

j_loop:
  %j = phi i32 [ 0, %j_loop_init ], [ %j_next, %j_next_block ]
  %j_cmp = icmp ult i32 %j, %n
  br i1 %j_cmp, label %k_loop_init, label %i_next_block

k_loop_init:
  %ij_offset = mul i32 %i, %n
  %ij_index = add i32 %ij_offset, %j
  %ij_index_i64 = zext i32 %ij_index to i64
  %C_ij_ptr = getelementptr double, ptr %C, i64 %ij_index_i64
  store double 0.0, ptr %C_ij_ptr
  br label %k_loop

k_loop:
  %k = phi i32 [ 0, %k_loop_init ], [ %k_next, %k_body ]
  %sum = phi double [ 0.0, %k_loop_init ], [ %new_sum, %k_body ]
  %k_cmp = icmp ult i32 %k, %n
  br i1 %k_cmp, label %k_body, label %k_done

k_body:
  %ik_offset = mul i32 %i, %n
  %ik_index = add i32 %ik_offset, %k
  %ik_index_i64 = zext i32 %ik_index to i64
  %A_ik_ptr = getelementptr double, ptr %A, i64 %ik_index_i64
  %A_ik = load double, ptr %A_ik_ptr
  
  %kj_offset = mul i32 %k, %n
  %kj_index = add i32 %kj_offset, %j
  %kj_index_i64 = zext i32 %kj_index to i64
  %B_kj_ptr = getelementptr double, ptr %B, i64 %kj_index_i64
  %B_kj = load double, ptr %B_kj_ptr
  
  %product = fmul double %A_ik, %B_kj
  %new_sum = fadd double %sum, %product
  
  %k_next = add i32 %k, 1
  br label %k_loop

k_done:
  store double %sum, ptr %C_ij_ptr
  %j_next = add i32 %j, 1
  br label %j_next_block

j_next_block:
  br label %j_loop

i_next_block:
  %i_next = add i32 %i, 1
  br label %i_loop

exit:
  ret void
}

; Matrix addition: C = A + B
define void @matrix_add(ptr %A, ptr %B, ptr %C, i32 %n) #3 {
entry:
  %n_i64 = zext i32 %n to i64
  %total_elements = mul i64 %n_i64, %n_i64
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i_next, %body ]
  %cmp = icmp ult i64 %i, %total_elements
  br i1 %cmp, label %body, label %exit

body:
  %A_ptr = getelementptr double, ptr %A, i64 %i
  %B_ptr = getelementptr double, ptr %B, i64 %i
  %C_ptr = getelementptr double, ptr %C, i64 %i
  
  %A_val = load double, ptr %A_ptr
  %B_val = load double, ptr %B_ptr
  %sum = fadd double %A_val, %B_val
  store double %sum, ptr %C_ptr
  
  %i_next = add i64 %i, 1
  br label %loop

exit:
  ret void
}

; Matrix subtraction: C = A - B
define void @matrix_subtract(ptr %A, ptr %B, ptr %C, i32 %n) #3 {
entry:
  %n_i64 = zext i32 %n to i64
  %total_elements = mul i64 %n_i64, %n_i64
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i_next, %body ]
  %cmp = icmp ult i64 %i, %total_elements
  br i1 %cmp, label %body, label %exit

body:
  %A_ptr = getelementptr double, ptr %A, i64 %i
  %B_ptr = getelementptr double, ptr %B, i64 %i
  %C_ptr = getelementptr double, ptr %C, i64 %i
  
  %A_val = load double, ptr %A_ptr
  %B_val = load double, ptr %B_ptr
  %diff = fsub double %A_val, %B_val
  store double %diff, ptr %C_ptr
  
  %i_next = add i64 %i, 1
  br label %loop

exit:
  ret void
}

; Extract quadrant from source matrix to destination
define void @extract_quadrant(ptr %src, ptr %dest, i32 %src_n, i32 %dest_n, i32 %row_offset, i32 %col_offset) #3 {
entry:
  br label %i_loop

i_loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %i_next_block ]
  %i_cmp = icmp ult i32 %i, %dest_n
  br i1 %i_cmp, label %j_loop_init, label %exit

j_loop_init:
  br label %j_loop

j_loop:
  %j = phi i32 [ 0, %j_loop_init ], [ %j_next, %copy_element ]
  %j_cmp = icmp ult i32 %j, %dest_n
  br i1 %j_cmp, label %copy_element, label %i_next_block

copy_element:
  ; Source index: (row_offset + i) * src_n + (col_offset + j)
  %src_row = add i32 %row_offset, %i
  %src_col = add i32 %col_offset, %j
  %src_row_offset = mul i32 %src_row, %src_n
  %src_index = add i32 %src_row_offset, %src_col
  %src_index_i64 = zext i32 %src_index to i64
  %src_ptr = getelementptr double, ptr %src, i64 %src_index_i64
  
  ; Destination index: i * dest_n + j
  %dest_row_offset = mul i32 %i, %dest_n
  %dest_index = add i32 %dest_row_offset, %j
  %dest_index_i64 = zext i32 %dest_index to i64
  %dest_ptr = getelementptr double, ptr %dest, i64 %dest_index_i64
  
  %value = load double, ptr %src_ptr
  store double %value, ptr %dest_ptr
  
  %j_next = add i32 %j, 1
  br label %j_loop

i_next_block:
  %i_next = add i32 %i, 1
  br label %i_loop

exit:
  ret void
}

; Insert quadrant from source to destination matrix
define void @insert_quadrant(ptr %src, ptr %dest, i32 %dest_n, i32 %src_n, i32 %row_offset, i32 %col_offset) #3 {
entry:
  br label %i_loop

i_loop:
  %i = phi i32 [ 0, %entry ], [ %i_next, %i_next_block ]
  %i_cmp = icmp ult i32 %i, %src_n
  br i1 %i_cmp, label %j_loop_init, label %exit

j_loop_init:
  br label %j_loop

j_loop:
  %j = phi i32 [ 0, %j_loop_init ], [ %j_next, %copy_element ]
  %j_cmp = icmp ult i32 %j, %src_n
  br i1 %j_cmp, label %copy_element, label %i_next_block

copy_element:
  ; Source index: i * src_n + j
  %src_row_offset = mul i32 %i, %src_n
  %src_index = add i32 %src_row_offset, %j
  %src_index_i64 = zext i32 %src_index to i64
  %src_ptr = getelementptr double, ptr %src, i64 %src_index_i64
  
  ; Destination index: (row_offset + i) * dest_n + (col_offset + j)
  %dest_row = add i32 %row_offset, %i
  %dest_col = add i32 %col_offset, %j
  %dest_row_offset = mul i32 %dest_row, %dest_n
  %dest_index = add i32 %dest_row_offset, %dest_col
  %dest_index_i64 = zext i32 %dest_index to i64
  %dest_ptr = getelementptr double, ptr %dest, i64 %dest_index_i64
  
  %value = load double, ptr %src_ptr
  store double %value, ptr %dest_ptr
  
  %j_next = add i32 %j, 1
  br label %j_loop

i_next_block:
  %i_next = add i32 %i, 1
  br label %i_loop

exit:
  ret void
}

; Function attributes
attributes #0 = { nofree nounwind }
attributes #1 = { nounwind }
attributes #2 = { nofree nounwind }
attributes #3 = { noinline nounwind optnone }