import hashlib
import json
import os

import torch


class GaLoreProjector:
    def __init__(
        self,
        rank,
        verbose=False,
        update_proj_gap=200,
        scale=1.0,
        proj_type="std",
        projector_index=None,
        projector_name=None,
        projector_save_dir=None,
        projector_load_dir=None,
    ):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type
        self.projector_index = projector_index
        self.projector_name = projector_name
        self.projector_save_dir = projector_save_dir
        self.projector_load_dir = projector_load_dir
        self.current_training_step = None
        self.last_projection_type = None
        self.last_refresh_details = None
        self.refresh_count = 0

    @staticmethod
    def _basis_sha256(basis):
        value = basis.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _basis_artifact_path(self, root, training_step):
        if self.projector_index is None or not self.projector_name:
            raise RuntimeError("Projector artifacts require stable parameter names and indices")
        return os.path.join(
            root,
            f"step_{int(training_step):06d}",
            f"parameter_{int(self.projector_index):04d}.pt",
        )

    @staticmethod
    def _append_manifest_sync(path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", buffering=1) as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _save_basis_artifact(
        self,
        basis,
        full_rank_grad,
        projection_type,
        refresh_index,
        source_kind,
        source_artifact=None,
    ):
        if not self.projector_save_dir or int(os.environ.get("RANK", "0")) != 0:
            return None
        if projection_type not in {"left", "right"}:
            raise RuntimeError("Projector artifact I/O currently supports left/right bases only")
        path = self._basis_artifact_path(
            self.projector_save_dir,
            self.current_training_step,
        )
        if os.path.exists(path):
            raise FileExistsError(f"Refusing to overwrite projector artifact: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cpu_basis = basis.detach().cpu().contiguous()
        basis_sha256 = self._basis_sha256(cpu_basis)
        metadata = {
            "format_version": 1,
            "global_step": int(self.current_training_step),
            "refresh_index": int(refresh_index),
            "parameter_name": self.projector_name,
            "parameter_index": int(self.projector_index),
            "full_rank_shape": list(full_rank_grad.shape),
            "basis_shape": list(cpu_basis.shape),
            "projection_orientation": projection_type,
            "projection_convention": (
                "low_rank_gradient=gradient@basis.T;backprojection=low_rank@basis"
                if projection_type == "right"
                else "low_rank_gradient=basis.T@gradient;backprojection=basis@low_rank"
            ),
            "proj_type": self.proj_type,
            "dtype": str(cpu_basis.dtype),
            "rank": int(self.rank),
            "scale": float(self.scale),
            "update_proj_gap": int(self.update_proj_gap),
            "source_kind": source_kind,
            "source_artifact": source_artifact,
            "basis_sha256": basis_sha256,
        }
        payload = {**metadata, "basis": cpu_basis}
        temporary_path = f"{path}.tmp.{os.getpid()}"
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
        relative_path = os.path.relpath(path, self.projector_save_dir)
        self._append_manifest_sync(
            os.path.join(self.projector_save_dir, "manifest.jsonl"),
            {**metadata, "artifact_path": relative_path},
        )
        return path

    def _load_basis_artifact(self, full_rank_grad, projection_type, refresh_index):
        path = self._basis_artifact_path(
            self.projector_load_dir,
            self.current_training_step,
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required healthy projector is missing: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        expected = {
            "format_version": 1,
            "global_step": int(self.current_training_step),
            "refresh_index": int(refresh_index),
            "parameter_name": self.projector_name,
            "parameter_index": int(self.projector_index),
            "full_rank_shape": list(full_rank_grad.shape),
            "projection_orientation": projection_type,
            "proj_type": self.proj_type,
            "rank": int(self.rank),
            "update_proj_gap": int(self.update_proj_gap),
        }
        mismatches = {
            key: (payload.get(key), value)
            for key, value in expected.items()
            if payload.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"External projector metadata mismatch at {path}: {mismatches}")
        basis = payload.get("basis")
        if not isinstance(basis, torch.Tensor):
            raise RuntimeError(f"External projector artifact has no tensor basis: {path}")
        if list(basis.shape) != payload.get("basis_shape"):
            raise RuntimeError(f"External projector basis shape metadata mismatch: {path}")
        if str(basis.dtype) != payload.get("dtype"):
            raise RuntimeError(f"External projector dtype metadata mismatch: {path}")
        actual_sha256 = self._basis_sha256(basis)
        if actual_sha256 != payload.get("basis_sha256"):
            raise RuntimeError(f"External projector checksum mismatch: {path}")
        if basis.dtype != full_rank_grad.dtype:
            raise RuntimeError(
                f"External projector dtype {basis.dtype} differs from gradient dtype "
                f"{full_rank_grad.dtype}: {path}"
            )
        return basis.to(full_rank_grad.device), path, actual_sha256

    def _refresh(self, full_rank_grad, projection_type):
        old_basis = self.ortho_matrix
        is_initial = old_basis is None
        refresh_index = self.refresh_count + 1
        if self.projector_load_dir:
            new_basis, source_artifact, source_sha256 = self._load_basis_artifact(
                full_rank_grad,
                projection_type,
                refresh_index,
            )
            source_kind = "healthy_external"
        else:
            new_basis = self.get_orthogonal_matrix(
                full_rank_grad,
                self.rank,
                projection_type,
            )
            source_artifact = None
            source_sha256 = self._basis_sha256(new_basis)
            source_kind = "computed_from_current_gradient"
        saved_artifact = self._save_basis_artifact(
            new_basis,
            full_rank_grad,
            projection_type,
            refresh_index,
            source_kind,
            source_artifact=source_artifact,
        )
        self.refresh_count += 1
        self.last_refresh_details = {
            "type": projection_type,
            "is_initial": is_initial,
            "refresh_index": self.refresh_count,
            "training_step": int(self.current_training_step),
            "old_basis_id": (
                None
                if is_initial
                else f"{self.projector_name}:refresh:{self.refresh_count - 1}"
            ),
            "new_basis_id": f"{self.projector_name}:refresh:{self.refresh_count}",
            "projector_source": source_kind,
            "source_artifact": source_artifact,
            "saved_artifact": saved_artifact,
            "basis_sha256": source_sha256,
        }
        return new_basis

    @staticmethod
    def _project_with_basis(full_rank_grad, basis, projection_type):
        basis = basis.to(full_rank_grad.device)
        if projection_type == "right":
            return full_rank_grad @ basis.t()
        if projection_type == "left":
            return basis.t() @ full_rank_grad
        raise ValueError("Projection comparison is only defined for left/right bases")

    def project_with_basis(self, full_rank_grad, basis, projection_type):
        """Project with a supplied basis without changing the installed basis."""
        if basis is None:
            raise ValueError("A non-initial basis is required for projection")
        return self._project_with_basis(full_rank_grad, basis, projection_type)

    def project_back_with_basis(self, low_rank_grad, basis, projection_type):
        """Backproject with a specified basis without mutating projector state."""
        if basis is None:
            raise ValueError("A non-initial basis is required for backprojection")
        basis = basis.to(low_rank_grad.device)
        if projection_type == "right":
            full_rank_grad = low_rank_grad @ basis
        elif projection_type == "left":
            full_rank_grad = basis @ low_rank_grad
        else:
            raise ValueError("Projection comparison is only defined for left/right bases")
        return full_rank_grad * self.scale

    def _should_refresh(self, iteration):
        return self.ortho_matrix is None or iteration % self.update_proj_gap == 0

    def _refresh_or_record_suppression(
        self,
        full_rank_grad,
        projection_type,
        suppress_scheduled_refresh,
    ):
        if not self._should_refresh(self.current_training_step):
            return
        if not suppress_scheduled_refresh:
            self.ortho_matrix = self._refresh(full_rank_grad, projection_type)
            return
        if self.ortho_matrix is None:
            raise RuntimeError("Cannot suppress the initial GaLore basis construction")

        # A frozen projector still has scheduled reset events. The installed
        # basis and refresh counter stay intact.
        current_basis_id = f"{self.projector_name}:refresh:{self.refresh_count}"
        scheduled_refresh_index = int(self.current_training_step) // int(self.update_proj_gap) + 1
        saved_artifact = self._save_basis_artifact(
            self.ortho_matrix,
            full_rank_grad,
            projection_type,
            scheduled_refresh_index,
            "frozen_reuse",
        )
        self.last_refresh_details = {
            "type": projection_type,
            "is_initial": False,
            "refresh_index": scheduled_refresh_index,
            "training_step": int(self.current_training_step),
            "old_basis_id": current_basis_id,
            "new_basis_id": current_basis_id,
            "actual_basis_refresh": False,
            "refresh_suppressed": True,
            "projector_source": "frozen_reuse",
            "saved_artifact": saved_artifact,
            "basis_sha256": self._basis_sha256(self.ortho_matrix),
        }

    def project(self, full_rank_grad, iteration, suppress_scheduled_refresh=False):
        self.current_training_step = int(iteration)
        self.last_refresh_details = None

        if self.proj_type == "std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                projection_type = "right"
                self._refresh_or_record_suppression(
                    full_rank_grad, projection_type, suppress_scheduled_refresh
                )
                low_rank_grad = full_rank_grad @ self.ortho_matrix.t().to(full_rank_grad.device)
            else:
                projection_type = "left"
                self._refresh_or_record_suppression(
                    full_rank_grad, projection_type, suppress_scheduled_refresh
                )
                low_rank_grad = self.ortho_matrix.t().to(full_rank_grad.device) @ full_rank_grad
        elif self.proj_type == "reverse_std":
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                projection_type = "left"
                self._refresh_or_record_suppression(
                    full_rank_grad, projection_type, suppress_scheduled_refresh
                )
                low_rank_grad = self.ortho_matrix.t().to(full_rank_grad.device) @ full_rank_grad
            else:
                projection_type = "right"
                self._refresh_or_record_suppression(
                    full_rank_grad, projection_type, suppress_scheduled_refresh
                )
                low_rank_grad = full_rank_grad @ self.ortho_matrix.t().to(full_rank_grad.device)
        elif self.proj_type == "right":
            projection_type = "right"
            self._refresh_or_record_suppression(
                full_rank_grad, projection_type, suppress_scheduled_refresh
            )
            low_rank_grad = full_rank_grad @ self.ortho_matrix.t().to(full_rank_grad.device)
        elif self.proj_type == "left":
            projection_type = "left"
            self._refresh_or_record_suppression(
                full_rank_grad, projection_type, suppress_scheduled_refresh
            )
            low_rank_grad = self.ortho_matrix.t().to(full_rank_grad.device) @ full_rank_grad
        elif self.proj_type == "full":
            projection_type = "full"
            self._refresh_or_record_suppression(
                full_rank_grad, projection_type, suppress_scheduled_refresh
            )
            low_rank_grad = (
                self.ortho_matrix[0].t().to(full_rank_grad.device)
                @ full_rank_grad
                @ self.ortho_matrix[1].t().to(full_rank_grad.device)
            )
        else:
            raise ValueError(f"Unsupported projection type: {self.proj_type}")

        self.last_projection_type = projection_type
        return low_rank_grad

    def project_back(self, low_rank_grad):
        if self.proj_type == "std":
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = low_rank_grad @ self.ortho_matrix.to(low_rank_grad.device)
            else:
                full_rank_grad = self.ortho_matrix.to(low_rank_grad.device) @ low_rank_grad
        elif self.proj_type == "reverse_std":
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]:
                full_rank_grad = self.ortho_matrix.to(low_rank_grad.device) @ low_rank_grad
            else:
                full_rank_grad = low_rank_grad @ self.ortho_matrix.to(low_rank_grad.device)
        elif self.proj_type == "right":
            full_rank_grad = low_rank_grad @ self.ortho_matrix.to(low_rank_grad.device)
        elif self.proj_type == "left":
            full_rank_grad = self.ortho_matrix.to(low_rank_grad.device) @ low_rank_grad
        elif self.proj_type == "full":
            full_rank_grad = (
                self.ortho_matrix[0].to(low_rank_grad.device)
                @ low_rank_grad
                @ self.ortho_matrix[1].to(low_rank_grad.device)
            )
        else:
            raise ValueError(f"Unsupported projection type: {self.proj_type}")
        return full_rank_grad * self.scale

    def get_orthogonal_matrix(self, weights, rank, projection_type):
        matrix = weights.detach().float()
        U, _, Vh = torch.linalg.svd(matrix, full_matrices=False)
        if projection_type == "right":
            result = Vh[:rank, :]
        elif projection_type == "left":
            result = U[:, :rank]
        elif projection_type == "full":
            return [
                U[:, :rank].to(device=weights.device, dtype=weights.dtype),
                Vh[:rank, :].to(device=weights.device, dtype=weights.dtype),
            ]
        else:
            raise ValueError("projection_type must be left, right, or full")
        return result.to(device=weights.device, dtype=weights.dtype)

