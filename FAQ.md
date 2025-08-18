# Frequently asked questions

## Error: program lacks permission to access the address space of the process

`nokozero` reads from the game process' memory to query the game state. This operation requires sufficient `ptrace` [permissions](https://www.kernel.org/doc/html/latest/admin-guide/LSM/Yama.html).

### Solution 1: Configure the Yama Linux Security Module

One approach is to globally disable `ptrace` restrictions.

```
echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope
```

Note: The `ptrace` permission level will reset to its default value upon restarting.

> [!WARNING]
> Running this command allows *any* process to read from and write to memory of any other process.

### Solution 2: Configure Linux capabilities

The `cap_sys_ptrace` capability allows programs to bypass `ptrace` restrictions for any process.

```
sudo setcap cap_sys_ptrace=eip <path to nokozero executable>
```

The capability persists until:
- manually removed with `sudo setcap -r <path to nokozero executable>`
- the program is recompiled or modified
- the program is copied without maintaining extended attributes
