from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil
from typing import Iterable, Sequence


class AccessKind(str, Enum):
    READ = "read"
    WRITE = "write"


@dataclass(frozen=True, slots=True)
class DramTiming:
    t_rcd: int = 15
    t_rp: int = 15
    t_cl: int = 17
    t_cwl: int = 14
    burst_cycles: int = 8
    write_recovery: int = 30
    transaction_overhead: int = 2

    @staticmethod
    def _ns_to_cycles(ns: float, ck_mhz: float) -> int:
        if ck_mhz <= 0:
            raise ValueError("ck_mhz must be positive")
        tck_ns = 1000.0 / ck_mhz
        return ceil(ns / tck_ns)

    @classmethod
    def lpddr5_x16_bg_from_ck(
        cls,
        ck_mhz: float,
        *,
        t_cl: int,
        t_cwl: int,
        burst_cycles: int = 8,
        transaction_overhead: int = 2,
        per_bank_precharge: bool = True,
    ) -> "DramTiming":
        """Build timing from the LPDDR5 x16 BG-mode table in the provided image.

        The visible rows in the screenshot are:
        - tRCD   = max(18ns, 2nCK)
        - tRPab  = max(21ns, 2nCK)
        - tRPpb  = max(18ns, 2nCK)
        - tWR    = max(34ns, 3nCK)

        This simplified model only tracks one `t_rp`, so by default it maps to
        single-bank precharge (`tRPpb`). Set `per_bank_precharge=False` to use
        all-bank precharge (`tRPab`) instead.

        The screenshot does not include tCL / tCWL, so they remain explicit.
        """

        t_rcd = max(cls._ns_to_cycles(18.0, ck_mhz), 2)
        t_rp_ns = 18.0 if per_bank_precharge else 21.0
        t_rp = max(cls._ns_to_cycles(t_rp_ns, ck_mhz), 2)
        write_recovery = max(cls._ns_to_cycles(34.0, ck_mhz), 3)
        return cls(
            t_rcd=t_rcd,
            t_rp=t_rp,
            t_cl=t_cl,
            t_cwl=t_cwl,
            burst_cycles=burst_cycles,
            write_recovery=write_recovery,
            transaction_overhead=transaction_overhead,
        )


@dataclass(frozen=True, slots=True)
class DramGeometry:
    row_buffer_bytes: int = 2048
    burst_bytes: int = 128
    request_bytes: int = 128


@dataclass(frozen=True, slots=True)
class Fragment:
    addr: int
    size: int

    def __post_init__(self) -> None:
        if self.addr < 0:
            raise ValueError("addr must be non-negative")
        if self.size <= 0:
            raise ValueError("size must be positive")


@dataclass(slots=True)
class ChunkReport:
    kind: AccessKind
    addr: int
    size: int
    row: int
    row_hit: bool
    start_cycle: float
    end_cycle: float
    setup_cycles: float
    column_cycles: float
    transfer_cycles: float
    partial_burst_cycles: float


@dataclass(slots=True)
class AccessReport:
    kind: AccessKind
    requested_bytes: int
    start_cycle: float
    end_cycle: float = 0.0
    total_cycles: float = 0.0
    request_count: int = 0
    data_cycles: float = 0.0
    setup_cycles: float = 0.0
    transaction_overhead_cycles: float = 0.0
    row_hit_count: int = 0
    row_miss_count: int = 0
    row_conflict_count: int = 0
    cross_row_cycles: float = 0.0
    fragmentation_cycles: float = 0.0
    partial_burst_cycles: float = 0.0
    chunks: list[ChunkReport] = field(default_factory=list)


@dataclass(slots=True)
class _BankState:
    open_row: int | None = None
    ready_cycle: float = 0.0


class SimpleDramLatencyModel:
    """A compact DRAM latency model focused on row switches and fragmentation.

    This intentionally keeps only the timing pieces that matter for a fast,
    software-side estimate:
    - row hit: column latency + data burst
    - closed row: ACT + column latency + data burst
    - row conflict / cross-row: PRE + ACT + column latency + data burst
    - fragmented access: repeated command overhead and partial-burst waste

    It does not model:
    - controller reordering
    - refresh stalls
    - bank-group timing
    - thermal / power feedback

    Notes:
    - accesses are first split into fixed-size DRAM transactions
    - the default parameters are aligned with DRAMsim3 LPDDR4 x16 2400 settings
    - the default `transaction_overhead` is calibrated to DRAMsim3 read latency
      under that LPDDR configuration
    - write latency here means service latency, not DRAMsim3's early write ack
    """

    def __init__(
        self,
        timing: DramTiming | None = None,
        geometry: DramGeometry | None = None,
    ) -> None:
        self.timing = timing or DramTiming()
        self.geometry = geometry or DramGeometry()
        self._bank = _BankState()
        self._cycle = 0.0

    @classmethod
    def lpddr4_x16_defaults(cls) -> "SimpleDramLatencyModel":
        """Defaults aligned with refs/DRAMsim3/configs/LPDDR4_8Gb_x16_2400.ini."""

        return cls(
            timing=DramTiming(
                t_rcd=15,
                t_rp=15,
                t_cl=17,
                t_cwl=14,
                burst_cycles=8,
                write_recovery=30,
                transaction_overhead=2,
            ),
            geometry=DramGeometry(
                row_buffer_bytes=2048,
                burst_bytes=128,
                request_bytes=128,
            ),
        )

    @classmethod
    def lpddr4_x16_lockstep_defaults(
        cls,
        stacked_die_count: int = 4,
    ) -> "SimpleDramLatencyModel":
        """LPDDR x16 defaults for lockstep stacked dies with shared latency."""

        if stacked_die_count <= 0:
            raise ValueError("stacked_die_count must be positive")

        base = cls.lpddr4_x16_defaults()
        return cls(
            timing=base.timing,
            geometry=DramGeometry(
                row_buffer_bytes=base.geometry.row_buffer_bytes * stacked_die_count,
                burst_bytes=base.geometry.burst_bytes * stacked_die_count,
                request_bytes=base.geometry.request_bytes * stacked_die_count,
            ),
        )

    @classmethod
    def hb_npu_hbm2_defaults(cls) -> "SimpleDramLatencyModel":
        """Defaults aligned with refs/HB-NPU-simulator/configs/HBM2_8Gb_x128.ini."""

        return cls(
            timing=DramTiming(
                t_rcd=14,
                t_rp=14,
                t_cl=14,
                t_cwl=4,
                burst_cycles=2,
                transaction_overhead=2,
            ),
            geometry=DramGeometry(
                row_buffer_bytes=2048,
                burst_bytes=32,
                request_bytes=32,
            ),
        )

    @classmethod
    def hb_npu_hbm2_lockstep_defaults(
        cls,
        stacked_die_count: int = 4,
    ) -> "SimpleDramLatencyModel":
        """HBM2 defaults for lockstep stacked dies with shared latency and wider data."""

        if stacked_die_count <= 0:
            raise ValueError("stacked_die_count must be positive")

        base = cls.hb_npu_hbm2_defaults()
        return cls(
            timing=base.timing,
            geometry=DramGeometry(
                row_buffer_bytes=base.geometry.row_buffer_bytes * stacked_die_count,
                burst_bytes=base.geometry.burst_bytes * stacked_die_count,
                request_bytes=base.geometry.request_bytes * stacked_die_count,
            ),
        )

    @property
    def cycle(self) -> float:
        return self._cycle

    def reset(self) -> None:
        self._bank = _BankState()
        self._cycle = 0.0

    def close_open_row(self) -> None:
        self._bank.open_row = None

    def read(
        self,
        addr: int | None = None,
        size: int | None = None,
        *,
        fragments: Sequence[tuple[int, int] | Fragment] | None = None,
        addresses: Sequence[int] | None = None,
        access_size: int | None = None,
    ) -> AccessReport:
        return self._access(
            AccessKind.READ,
            addr,
            size,
            fragments,
            addresses,
            access_size,
        )

    def write(
        self,
        addr: int | None = None,
        size: int | None = None,
        *,
        fragments: Sequence[tuple[int, int] | Fragment] | None = None,
        addresses: Sequence[int] | None = None,
        access_size: int | None = None,
    ) -> AccessReport:
        return self._access(
            AccessKind.WRITE,
            addr,
            size,
            fragments,
            addresses,
            access_size,
        )

    def read_addresses(
        self,
        addresses: Sequence[int],
        access_size: int | None = None,
    ) -> AccessReport:
        return self.read(addresses=addresses, access_size=access_size)

    def write_addresses(
        self,
        addresses: Sequence[int],
        access_size: int | None = None,
    ) -> AccessReport:
        return self.write(addresses=addresses, access_size=access_size)

    def simulate(
        self,
        requests: Iterable[tuple[AccessKind, int, int]],
    ) -> list[AccessReport]:
        reports: list[AccessReport] = []
        for kind, addr, size in requests:
            reports.append(self._access(kind, addr, size, None))
        return reports

    def _access(
        self,
        kind: AccessKind,
        addr: int | None,
        size: int | None,
        fragments: Sequence[tuple[int, int] | Fragment] | None,
        addresses: Sequence[int] | None,
        access_size: int | None,
    ) -> AccessReport:
        fragment_list = self._normalize_fragments(
            addr,
            size,
            fragments,
            addresses,
            access_size,
        )
        report = AccessReport(
            kind=kind,
            requested_bytes=sum(fragment.size for fragment in fragment_list),
            start_cycle=self._cycle,
        )

        for fragment_index, fragment in enumerate(fragment_list):
            if fragment_index > 0:
                report.fragmentation_cycles += self._column_cycles(kind)

            request_chunks = list(self._split_by_request(fragment.addr, fragment.size))
            for request_index, (request_addr, request_size) in enumerate(request_chunks):
                if fragment_index > 0 and request_index == 0:
                    partial_penalty = self._partial_burst_penalty(request_size)
                    report.fragmentation_cycles += partial_penalty

                report.request_count += 1
                report.transaction_overhead_cycles += self.timing.transaction_overhead
                self._cycle += self.timing.transaction_overhead

                for chunk_addr, chunk_size in self._split_by_row(request_addr, request_size):
                    self._process_chunk(kind, chunk_addr, chunk_size, report)

        report.end_cycle = self._cycle
        report.total_cycles = report.end_cycle - report.start_cycle
        return report

    def _process_chunk(
        self,
        kind: AccessKind,
        addr: int,
        size: int,
        report: AccessReport,
    ) -> None:
        row = addr // self.geometry.row_buffer_bytes
        start_cycle = max(self._cycle, self._bank.ready_cycle)

        setup_cycles = 0.0
        row_hit = self._bank.open_row == row
        if row_hit:
            report.row_hit_count += 1
        elif self._bank.open_row is None:
            setup_cycles = float(self.timing.t_rcd)
            report.row_miss_count += 1
        else:
            setup_cycles = float(self.timing.t_rp + self.timing.t_rcd)
            report.row_conflict_count += 1
            report.cross_row_cycles += setup_cycles

        bursts = ceil(size / self.geometry.burst_bytes)
        transfer_cycles = float(bursts * self.timing.burst_cycles)
        column_cycles = float(self._column_cycles(kind))
        partial_burst_cycles = self._partial_burst_penalty(size)
        end_cycle = start_cycle + setup_cycles + column_cycles + transfer_cycles

        report.setup_cycles += setup_cycles
        report.data_cycles += column_cycles + transfer_cycles
        report.partial_burst_cycles += partial_burst_cycles
        report.chunks.append(
            ChunkReport(
                kind=kind,
                addr=addr,
                size=size,
                row=row,
                row_hit=row_hit,
                start_cycle=start_cycle,
                end_cycle=end_cycle,
                setup_cycles=setup_cycles,
                column_cycles=column_cycles,
                transfer_cycles=transfer_cycles,
                partial_burst_cycles=partial_burst_cycles,
            )
        )

        self._bank.open_row = row
        self._bank.ready_cycle = end_cycle + (
            self.timing.write_recovery if kind is AccessKind.WRITE else 0.0
        )
        self._cycle = end_cycle

    def _normalize_fragments(
        self,
        addr: int | None,
        size: int | None,
        fragments: Sequence[tuple[int, int] | Fragment] | None,
        addresses: Sequence[int] | None,
        access_size: int | None,
    ) -> list[Fragment]:
        modes_enabled = int(fragments is not None) + int(addresses is not None)
        modes_enabled += int(addr is not None or size is not None)
        if modes_enabled != 1:
            raise ValueError(
                "provide exactly one access description: "
                "(addr and size) or addresses or fragments"
            )

        if fragments is None and addresses is None:
            if addr is None or size is None:
                raise ValueError("addr and size must be provided together")
            if access_size is not None:
                raise ValueError("access_size is only valid with addresses")
            return [Fragment(addr=addr, size=size)]

        if addresses is not None:
            if fragments is not None:
                raise ValueError("addresses and fragments cannot be used together")
            if addr is not None or size is not None:
                raise ValueError("addresses cannot be mixed with addr/size")
            if not addresses:
                raise ValueError("addresses must not be empty")
            fragment_size = access_size or self.geometry.burst_bytes
            normalized: list[Fragment] = []
            for fragment_addr in addresses:
                fragment = Fragment(addr=fragment_addr, size=fragment_size)
                if not normalized:
                    normalized.append(fragment)
                    continue

                last = normalized[-1]
                last_end = last.addr + last.size
                if fragment.addr < last_end:
                    raise ValueError(
                        "addresses must be strictly increasing and non-overlapping"
                    )
                if fragment.addr == last_end:
                    normalized[-1] = Fragment(addr=last.addr, size=last.size + fragment.size)
                else:
                    normalized.append(fragment)
            return normalized

        if access_size is not None:
            raise ValueError("access_size is only valid with addresses")

        normalized: list[Fragment] = []
        for fragment in fragments:
            if isinstance(fragment, Fragment):
                normalized.append(fragment)
            else:
                normalized.append(Fragment(addr=fragment[0], size=fragment[1]))
        if not normalized:
            raise ValueError("fragments must not be empty")
        return normalized

    def _split_by_row(self, addr: int, size: int) -> Iterable[tuple[int, int]]:
        remaining = size
        current = addr
        row_bytes = self.geometry.row_buffer_bytes
        while remaining > 0:
            offset_in_row = current % row_bytes
            chunk_size = min(remaining, row_bytes - offset_in_row)
            yield current, chunk_size
            current += chunk_size
            remaining -= chunk_size

    def _split_by_request(self, addr: int, size: int) -> Iterable[tuple[int, int]]:
        remaining = size
        current = addr
        request_bytes = self.geometry.request_bytes
        while remaining > 0:
            chunk_size = min(remaining, request_bytes)
            yield current, chunk_size
            current += chunk_size
            remaining -= chunk_size

    def _column_cycles(self, kind: AccessKind) -> int:
        return self.timing.t_cl if kind is AccessKind.READ else self.timing.t_cwl

    def _partial_burst_penalty(self, size: int) -> float:
        burst_bytes = self.geometry.burst_bytes
        remainder = size % burst_bytes
        if remainder == 0:
            return 0.0
        wasted_ratio = (burst_bytes - remainder) / burst_bytes
        return wasted_ratio * self.timing.burst_cycles
