

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a b)
(on b e)
(on c d)
(on d a)
(ontable e)
(ontable f)
(clear c)
(clear f)
)
(:goal
(and
(on b d)
(on d c)
(on e b))
)
)


